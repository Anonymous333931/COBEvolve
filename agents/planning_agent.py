# ── FILE: cobol_moderniser/agents/planning_agent.py ──────────────
"""
agents/planning_agent.py — MAPE-K planning controller.

ANALYZE phase of the MAPE-K loop. This module is controller logic, not one of
the six paper-facing COBEvolve agents.

Phases:
  1. RVF Scoring    — risk, value, feasibility per module
  2. Ordering       — topological + composite score sort
  3. Serialise      — write MigrationPlan to JSON
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from config import config
from utils.graph_utils import (
    get_leaf_programs, detect_cycles, topological_order
)

logger = logging.getLogger(__name__)

# ── Scoring constants (tune for your codebase) ──────────────
MAX_DEGREE      = 10   # fan_in + fan_out above this = max risk
MAX_COMPLEXITY  = 50   # paragraphs + data_items + rules*2 above this = max value
RULE_PENALTY    = 0.10 # feasibility drops 0.10 per undocumented rule
MIN_FEASIBILITY = 0.20 # floor — even the hardest module has 20% feasibility

# ── Data models ──────────────────────────────────────────
@dataclass
class ScoredModule:
    program_id:   str
    filepath:     str
    priority:     int   = 0
    risk_score:   float = 0.0
    value_score:  float = 0.0
    feasibility:  float = 1.0
    composite:    float = 0.0
    reason:       str   = ''
    depends_on:   list  = field(default_factory=list)
    called_by:    list  = field(default_factory=list)
    is_leaf:      bool  = False
    has_cycles:   bool  = False

@dataclass
class MigrationPlan:
    modules:        list[ScoredModule]
    cycles:         list[list[str]] = field(default_factory=list)
    total_programs: int             = 0
    leaf_count:     int             = 0
    generated_at:   str             = ''

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()
        self.total_programs = len(self.modules)
        self.leaf_count     = sum(1 for m in self.modules if m.is_leaf)


# ══════════════════════════════════════════════════════
# PHASE 1: RVF SCORING
# ══════════════════════════════════════════════════════

def _compute_risk(prog_info: dict, graph: nx.DiGraph,
                  cyclic_nodes: set[str]) -> float:
    """
    Risk = coupling density of the module.
    High fan_in (many callers) = dangerous to move.
    High fan_out (calls many others) = many dependencies to resolve.
    Cyclic dependency = automatic risk ceiling of 0.9.
    """
    program_id = prog_info.get('program_id', '')
    fan_in     = prog_info.get('fan_in',  0)
    fan_out    = prog_info.get('fan_out', 0)
    degree     = fan_in + fan_out
    base_risk  = min(1.0, degree / MAX_DEGREE)
    if program_id in cyclic_nodes:
        return max(base_risk, 0.9)   # cycles are very risky
    return base_risk


def _compute_value(prog_info: dict) -> float:
    """
    Value = modernisation reward for translating this module.
    More paragraphs + data items + business rules = more to gain.
    """
    paragraphs  = len(prog_info.get('paragraphs', []))
    data_items  = len(prog_info.get('data_items',  []))
    rules       = len(prog_info.get('business_rules', []))
    complexity  = paragraphs + data_items + (rules * 2)
    return min(1.0, complexity / MAX_COMPLEXITY)


def _compute_feasibility(prog_info: dict) -> float:
    """
    Feasibility = how easy translation will be.
    Many undocumented rules make the LLM translation harder.
    Floor at MIN_FEASIBILITY so no module is rated impossible.
    """
    rules      = len(prog_info.get('business_rules', []))
    penalty    = rules * RULE_PENALTY
    return max(MIN_FEASIBILITY, 1.0 - penalty)


def score_module(
    program_id: str,
    prog_info:  dict,
    graph:      nx.DiGraph,
    leaf_set:   set[str],
    cyclic_nodes: set[str],
) -> ScoredModule:
    """
    Compute Risk, Value, Feasibility, and Composite for one module.
    Returns a ScoredModule (priority assigned later in Phase 2).
    """
    info = dict(prog_info)
    info['program_id'] = program_id

    risk        = _compute_risk(info, graph, cyclic_nodes)
    value       = _compute_value(info)
    feasibility = _compute_feasibility(info)
    composite   = (1.0 - risk) * value * feasibility

    # Build plain-English reason
    parts = []
    if info.get('is_leaf', False):
        parts.append('leaf node (no downstream dependencies)')
    if risk < 0.3:
        parts.append('low coupling')
    elif risk > 0.7:
        parts.append('high coupling — migrate carefully')
    if program_id in cyclic_nodes:
        parts.append('CYCLIC dependency detected — flag for human review')
    if value > 0.6:
        parts.append('high modernisation value')
    if feasibility < 0.5:
        parts.append('complex business rules — LLM may need guidance')
    reason = '; '.join(parts) or 'standard module'

    # Dependency lists
    depends_on = [v for _, v in graph.out_edges(program_id)
                  if v != program_id]
    called_by  = [u for u, _ in graph.in_edges(program_id)
                  if u != program_id]

    return ScoredModule(
        program_id  = program_id,
        filepath    = prog_info.get('filepath', ''),
        risk_score  = round(risk, 4),
        value_score = round(value, 4),
        feasibility = round(feasibility, 4),
        composite   = round(composite, 4),
        reason      = reason,
        depends_on  = depends_on,
        called_by   = called_by,
        is_leaf     = program_id in leaf_set,
        has_cycles  = program_id in cyclic_nodes,
    )



# ══════════════════════════════════════════════════════
# PHASE 2: DEPENDENCY-AWARE ORDERING
# ══════════════════════════════════════════════════════

def order_modules(
    scored: list[ScoredModule],
    graph:  nx.DiGraph,
) -> list[ScoredModule]:
    """
    Order modules so that dependencies are migrated before the modules
    that depend on them. Within the same topological level, sort by
    composite score descending (highest value first).

    Strategy:
      1. Topological sort gives a safe dependency-first order.
      2. Within the same topo-level, sort by composite score desc.
      3. Modules with cycles are pinned to the END of the plan
         and flagged has_cycles=True for human review.
    """
    by_id = {m.program_id: m for m in scored}

    # Separate cyclic from acyclic modules
    cyclic_ids = {m.program_id for m in scored if m.has_cycles}
    acyclic    = [m for m in scored if not m.has_cycles]
    cyclic     = [m for m in scored if m.has_cycles]

    # Topological order for acyclic subset
    real_nodes = [n for n in graph.nodes()
                  if graph.nodes[n].get('filepath') != 'external'
                  and n not in cyclic_ids]
    sub_graph  = graph.subgraph(real_nodes).copy()
    sub_graph.remove_edges_from(nx.selfloop_edges(sub_graph))

    try:
        topo = list(reversed(list(nx.topological_sort(sub_graph))))
    except nx.NetworkXUnfeasible:
        logger.warning('Unexpected cycle in acyclic subgraph; using degree sort')
        topo = sorted(real_nodes,
                      key=lambda n: graph.nodes[n].get('fan_out', 0))

    # Build topo-position lookup
    topo_pos = {pid: i for i, pid in enumerate(topo)}

    # Sort: primary = topo position, secondary = composite desc
    acyclic.sort(key=lambda m: (
        topo_pos.get(m.program_id, 9999),
        -m.composite
    ))

    # Cyclic modules go at the END, sorted by risk desc (review riskiest first)
    cyclic.sort(key=lambda m: -m.risk_score)

    ordered = acyclic + cyclic

    # Assign 1-based priority
    for i, mod in enumerate(ordered, 1):
        mod.priority = i

    return ordered


# ══════════════════════════════════════════════════════
# PHASE 3: SERIALISE MIGRATION PLAN
# ══════════════════════════════════════════════════════

def serialise_plan(plan: MigrationPlan, output_path: str | None = None) -> str:
    """
    Convert MigrationPlan to JSON and optionally write to disk.
    Returns the JSON string.
    """
    def _module_to_dict(m: ScoredModule) -> dict:
        return {
            'program_id':  m.program_id,
            'filepath':    m.filepath,
            'priority':    m.priority,
            'risk_score':  m.risk_score,
            'value_score': m.value_score,
            'feasibility': m.feasibility,
            'composite':   m.composite,
            'reason':      m.reason,
            'depends_on':  m.depends_on,
            'called_by':   m.called_by,
            'is_leaf':     m.is_leaf,
            'has_cycles':  m.has_cycles,
        }

    payload = {
        'modules':        [_module_to_dict(m) for m in plan.modules],
        'cycles':         plan.cycles,
        'total_programs': plan.total_programs,
        'leaf_count':     plan.leaf_count,
        'generated_at':   plan.generated_at,
    }
    json_str = json.dumps(payload, indent=2)

    out = output_path or str(
        Path(config.OUTPUT_DIR) / 'migration_plan.json'
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json_str, encoding='utf-8')
    logger.info('Migration plan written to %s', out)
    return json_str


def load_plan(json_path: str | Path) -> MigrationPlan:
    """Load a previously saved MigrationPlan from JSON."""
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    modules = [
        ScoredModule(
            program_id=m.get('program_id', ''),
            filepath=m.get('filepath', ''),
            priority=m.get('priority', 0),
            risk_score=m.get('risk_score', 0.0),
            value_score=m.get('value_score', 0.0),
            feasibility=m.get('feasibility', 1.0),
            composite=m.get('composite', 0.0),
            reason=m.get('reason', ''),
            depends_on=m.get('depends_on', []),
            called_by=m.get('called_by', []),
            is_leaf=m.get('is_leaf', False),
            has_cycles=m.get('has_cycles', False),
        )
        for m in data['modules']
    ]
    return MigrationPlan(
        modules        = modules,
        cycles         = data.get('cycles', []),
        total_programs = data.get('total_programs', len(modules)),
        leaf_count     = data.get('leaf_count', 0),
        generated_at   = data.get('generated_at', ''),
    )

# ══════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════

def plan(
    knowledge_graph: dict[str, Any],
    output_path:     str | None = None,
) -> MigrationPlan:
    """
    Main MAPE-K planning entry point.
    Call from the Orchestrator (Step 7) or directly for testing.

    Parameters
    ----------
    knowledge_graph : dict produced by Comprehension Agent (Step 2)
    output_path     : where to write migration_plan.json (optional)

    Returns
    -------
    MigrationPlan consumed by Translation Agent (Step 4)
    """
    config.setup()
    logger.info('=== MAPE-K planning starting ==='  )

    programs = knowledge_graph.get('programs', {})
    graph    = knowledge_graph.get('_graph')

    if not programs:
        raise ValueError('knowledge_graph contains no programs')
    if graph is None:
        raise ValueError('knowledge_graph missing _graph key')

    # Compute helper sets
    leaf_set     = set(get_leaf_programs(graph))
    raw_cycles   = detect_cycles(graph)
    cyclic_nodes = {n for cycle in raw_cycles for n in cycle
                    if len(cycle) > 1}

    if cyclic_nodes:
        logger.warning('Cyclic dependencies detected in: %s', cyclic_nodes)

    # Phase 1: Score every module
    scored = []
    for pid, pinfo in programs.items():
        pinfo_with_leaf = dict(pinfo)
        pinfo_with_leaf['is_leaf'] = pid in leaf_set
        sm = score_module(pid, pinfo_with_leaf, graph, leaf_set, cyclic_nodes)
        scored.append(sm)
        logger.debug('  %s: risk=%.2f val=%.2f feas=%.2f comp=%.2f',
                     pid, sm.risk_score, sm.value_score,
                     sm.feasibility, sm.composite)

    # Phase 2: Order modules
    ordered = order_modules(scored, graph)

    # Phase 3: Build plan and serialise
    migration_plan = MigrationPlan(
        modules  = ordered,
        cycles   = [c for c in raw_cycles if len(c) > 1],
    )
    serialise_plan(migration_plan, output_path)

    logger.info('=== MAPE-K planning done: %d modules, %d leaves, %d cycles ===',
                migration_plan.total_programs,
                migration_plan.leaf_count,
                len(migration_plan.cycles))
    return migration_plan


def summarise_plan(migration_plan: MigrationPlan) -> str:
    """Human-readable summary of the migration plan."""
    lines = [
        f'Migration Plan: {migration_plan.total_programs} modules, '
        f'{migration_plan.leaf_count} leaves'
    ]
    if migration_plan.cycles:
        lines.append(f'  WARNING: {len(migration_plan.cycles)} circular '
                     f'dependency cycle(s) detected — flagged for human review')
    lines.append('')
    lines.append(f'  {"Priority":<8} {"Program":<14} '
                 f'{"Risk":>6} {"Value":>6} {"Feas":>6} '
                 f'{"Score":>7}  Reason')
    lines.append('  ' + '-'*80)
    for m in migration_plan.modules:
        cycle_flag = ' [CYCLE]' if m.has_cycles else ''
        lines.append(
            f'  {m.priority:<8} {m.program_id:<14} '
            f'{m.risk_score:>6.2f} {m.value_score:>6.2f} '
            f'{m.feasibility:>6.2f} {m.composite:>7.4f}  '
            f'{m.reason[:40]}{cycle_flag}'
        )
    return '\n'.join(lines)
