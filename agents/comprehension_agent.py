# ── FILE: cobol_moderniser/agents/comprehension_agent.py ──────────
"""
agents/comprehension_agent.py -- Comprehension Agent
MONITOR phase of the MAPE-K loop.


Grounded in: A-COBREX (ICSE 2025), COBRAIN, COBREX (ICSME 2022)


Phases:
  1. Scan     -- find and parse all COBOL files
  2. Graph    -- build NetworkX dependency graph
  3. Rules    -- extract business rules via Ollama
  4. Assemble -- combine into knowledge_graph dict
"""


from __future__ import annotations


import json
import logging
from pathlib import Path
from typing import Any


from config import config
from utils.cobol_parser import (
    COBOLProgram, parse_cobol_file, parse_cobol_directory
)
from utils.graph_utils import (
    build_dependency_graph, graph_summary, get_leaf_programs
)
from utils.ollama_client import ollama_client, OllamaError


logger = logging.getLogger(__name__)


class ComprehendResult(tuple):
    """Tuple-compatible result that also behaves like the kg dict."""

    def __new__(cls, kg: dict[str, Any], graph):
        return super().__new__(cls, (kg, graph))

    @property
    def kg(self) -> dict[str, Any]:
        return tuple.__getitem__(self, 0)

    @property
    def graph(self):
        return tuple.__getitem__(self, 1)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.kg[key]
        return tuple.__getitem__(self, key)

    def get(self, key, default=None):
        return self.kg.get(key, default)

    def keys(self):
        return self.kg.keys()

    def items(self):
        return self.kg.items()

    def values(self):
        return self.kg.values()

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.kg
        return tuple.__contains__(self, item)




# ─────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────


_BUSINESS_RULE_SYSTEM = (
    'You are a senior COBOL business analyst with 30 years of mainframe '
    'experience. You extract precise business rules from COBOL source code. '
    'You always respond with valid JSON only -- no markdown, no preamble.'
)


_BUSINESS_RULE_PROMPT = '''
Analyse the COBOL program below and extract every distinct business rule.


A business rule is a specific condition, calculation, validation, or
decision that implements business logic (NOT technical implementation).


Examples of business rules:
  - 'Transaction amount must not exceed 50,000'
  - 'Account balance after debit must remain above minimum balance of 100'
  - 'Employees working over 160 hours receive 1.5x overtime pay'


Return ONLY this JSON array (no other text):
[
  {{
    "rule": "plain English description of the business rule",
    "paragraph": "COBOL paragraph name where implemented",
    "variables": ["WS-BALANCE", "WS-MIN-BALANCE"],
    "confidence": 0.95
  }}
]


PROGRAM-ID: {program_id}


DATA ITEMS:
{data_items}


PROCEDURE DIVISION:
{procedure}
'''




_SUMMARY_PROMPT = '''
In 2-3 sentences, summarise what this COBOL program does from a
BUSINESS perspective (not technical). Focus on what business problem
it solves and what data it processes.


PROGRAM-ID: {program_id}
PARAGRAPHS: {paragraphs}
DATA ITEMS: {data_items}
'''





# ═════════════════════════════════════════════════════════════════
# PHASE 1: SCAN
# ═════════════════════════════════════════════════════════════════


def scan_cobol_files(
    source: str | Path,
    repo_root: str | Path | None = None,
) -> list[COBOLProgram]:
    """
    Find and parse all COBOL files from source.
    source can be:
      - Path to a single .cbl file
      - Path to a directory (searches recursively)
      - A list of COBOLProgram objects (pass-through)
    """
    source = Path(str(source)) if not isinstance(source, list) else source


    if isinstance(source, list):
        programs = source
    elif source.is_dir():
        logger.info('Scanning directory: %s', source)
        programs = parse_cobol_directory(str(source), repo_root=repo_root or source)
    elif source.is_file():
        logger.info('Parsing single file: %s', source)
        programs = [parse_cobol_file(str(source))]
    else:
        raise FileNotFoundError(f'Source not found: {source}')


    if not programs:
        raise ValueError(f'No COBOL files found in: {source}')


    logger.info('Scanned %d COBOL program(s)', len(programs))
    for prog in programs:
        logger.debug('  %s: %d paragraphs, %d data items',
                     prog.program_id, len(prog.paragraphs),
                     len(prog.all_data_items))
    return programs

# ═════════════════════════════════════════════════════════════════
# PHASE 2: DEPENDENCY GRAPH
# ═════════════════════════════════════════════════════════════════


def build_graph(programs: list[COBOLProgram]):
    """Build and log the dependency graph. Returns NetworkX DiGraph."""
    G = build_dependency_graph(programs)
    logger.info(graph_summary(G))
    leaves = get_leaf_programs(G)
    if leaves:
        logger.info('Leaf programs (safest to migrate first): %s', leaves)
    return G
# ═════════════════════════════════════════════════════════════════
# PHASE 3: BUSINESS RULE EXTRACTION
# ═════════════════════════════════════════════════════════════════


def _format_data_items(program: COBOLProgram) -> str:
    """Format data items as a compact text block for the LLM prompt."""
    lines = []
    for item in program.all_data_items[:30]:  # cap at 30 to save tokens
        val_str = f" VALUE {item.value}" if item.value else ""
        pic_str = f" PIC {item.pic}" if item.pic else ""
        lines.append(f"  {item.level:02d} {item.name}{pic_str}{val_str}")
    return "\n".join(lines) or "  (none)"




def _format_procedure(program: COBOLProgram) -> str:
    """Format procedure division as text block for the LLM prompt."""
    sections = []
    for para in program.paragraphs:
        sections.append(f"{para.name}.\n{para.raw_body.strip()}")
    text = "\n\n".join(sections)
    # Cap at 5000 chars to stay within context window
    return text[:5000] + ("\n...[truncated]" if len(text) > 5000 else "")




def extract_business_rules(program: COBOLProgram) -> list[dict]:
    """
    Call Ollama to extract business rules from a COBOL program.
    Returns list of {rule, paragraph, variables, confidence} dicts.
 Falls back to empty list on any failure (pipeline continues).
    """
    if not program.paragraphs:
        return []


    prompt = _BUSINESS_RULE_PROMPT.format(
        program_id=program.program_id,
        data_items=_format_data_items(program),
        procedure=_format_procedure(program),
    )


    try:
        result = ollama_client.generate_json(
            model=config.MODEL_ANALYSIS,
            prompt=prompt,
            system=_BUSINESS_RULE_SYSTEM,
            fallback=[],
        )
        if not isinstance(result, list):
            logger.warning('Rule extraction returned non-list for %s', program.program_id)
            return []
        # Validate each rule has required fields
        valid = []
        for r in result:
            if isinstance(r, dict) and 'rule' in r:
                valid.append({
                    'rule':       r.get('rule', ''),
                    'paragraph':  r.get('paragraph', 'UNKNOWN'),
                    'variables':  r.get('variables', []),
                    'confidence': float(r.get('confidence', 0.5)),
                })
        logger.info('Extracted %d business rules from %s',
                    len(valid), program.program_id)
        return valid
    except OllamaError as exc:
        logger.warning('Ollama unavailable for %s: %s', program.program_id, exc)
        return []
    except Exception as exc:
        logger.warning('Rule extraction failed for %s: %s', program.program_id, exc)
        return []




def get_program_summary(program: COBOLProgram) -> str:
    """
    Get a short business-level summary of what this program does.
    Used in the knowledge graph for human-readable documentation.
    """
    prompt = _SUMMARY_PROMPT.format(
        program_id=program.program_id,
        paragraphs=', '.join(program.paragraph_names[:10]),
        data_items=_format_data_items(program),
    )
    try:
        return ollama_client.ask_analysis(prompt)[:500]
    except Exception:
        return f'{program.program_id}: summary unavailable'

# ═════════════════════════════════════════════════════════════════
# PHASE 4: ASSEMBLE KNOWLEDGE GRAPH
# ═════════════════════════════════════════════════════════════════


def _program_to_dict(prog: COBOLProgram, graph) -> dict:
    """Serialise one COBOLProgram to a JSON-safe dict for the knowledge graph."""
    data_items = [
        {
            'name':        item.name,
            'python_name': item.python_name,
            'type':        item.python_type,
            'pic':         item.pic or '',
            'level':       item.level,
            'value':       item.value or '',
        }
        for item in prog.all_data_items
    ]
    calls_external = list({
        call
        for para in prog.paragraphs
        for call in para.calls
    })
    node_data = graph.nodes.get(prog.program_id, {})
    return {
        'filepath':       prog.filepath,
        'paragraphs':     prog.paragraph_names,
        'data_items':     data_items,
        'business_rules': [],        # filled by extract_business_rules()
        'summary':        '',         # filled by get_program_summary()
        'calls_external': calls_external,
        'fan_in':         node_data.get('fan_in', 0),
        'fan_out':        node_data.get('fan_out', 0),
        'raw_source':     prog.raw_source,
    }




def assemble_knowledge_graph(
    programs: list[COBOLProgram],
    graph,
    extract_rules: bool = True,
    extract_summaries: bool = False,
) -> dict[str, Any]:
    """
    Combine programs + graph + business rules into the knowledge_graph dict.


    Parameters
    ----------
    programs         : parsed COBOL programs from Phase 1
    graph            : NetworkX DiGraph from Phase 2
    extract_rules    : if True, call Ollama for each program (takes time)
    extract_summaries: if True, also generate business summaries (slower)
    """
    kg: dict[str, Any] = {'programs': {}, 'edges': [], '_graph': graph}


    total = len(programs)
    for i, prog in enumerate(programs, 1):
        logger.info('[%d/%d] Processing %s ...', i, total, prog.program_id)


        prog_dict = _program_to_dict(prog, graph)


        # Phase 3: Business rules via Ollama (optional)
        if extract_rules:
            prog_dict['business_rules'] = extract_business_rules(prog)


        # Optional: business-level summary
        if extract_summaries:
            prog_dict['summary'] = get_program_summary(prog)


        kg['programs'][prog.program_id] = prog_dict


    # Edges
    for u, v, data in graph.edges(data=True):
        kg['edges'].append({
            'from': u,
            'to':   v,
            'type': data.get('type', 'UNKNOWN'),
        })


    logger.info('Knowledge graph assembled: %d programs, %d edges',
                len(kg['programs']), len(kg['edges']))
    return kg




# ═════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════


def comprehend(
    source: str | Path | list[COBOLProgram],
    extract_rules: bool = True,
    extract_summaries: bool = False,
    repo_root: str | Path | None = None,
) -> ComprehendResult:
    """
    Main Comprehension Agent entry point.


    Call this from the Orchestrator (Step 7) or directly for testing.


    Parameters
    ----------
    source           : .cbl file path, directory path, or pre-parsed list
    extract_rules    : call Ollama for business rule extraction (default True)
    extract_summaries: generate plain-English program summaries (default False)


    Returns
    -------
    tuple-like result: (knowledge_graph dict, NetworkX graph)
    """
    config.setup()
    logger.info('=== Comprehension Agent starting ===')


    # Phase 1: Scan
    programs = scan_cobol_files(source, repo_root=repo_root)


    # Phase 2: Build graph
    graph = build_graph(programs)


    # Phases 3 + 4: Extract rules and assemble
    kg = assemble_knowledge_graph(
        programs, graph,
        extract_rules=extract_rules,
        extract_summaries=extract_summaries,
    )


    logger.info('=== Comprehension Agent done ===')
    kg_without_graph = {k: v for k, v in kg.items() if k != '_graph'}
    return ComprehendResult(kg_without_graph, graph)


def comprehend_multiple_repos(
    repo_paths: list[str | Path],
    extract_rules: bool = True,
    extract_summaries: bool = False,
) -> ComprehendResult:
    """Comprehend multiple repo roots as one combined knowledge graph."""
    programs: list[COBOLProgram] = []
    for repo_path in repo_paths:
        programs.extend(scan_cobol_files(repo_path, repo_root=repo_path))
    graph = build_graph(programs)
    kg = assemble_knowledge_graph(
        programs, graph,
        extract_rules=extract_rules,
        extract_summaries=extract_summaries,
    )
    kg_without_graph = {k: v for k, v in kg.items() if k != '_graph'}
    return ComprehendResult(kg_without_graph, graph)




def summarise_knowledge_graph(kg: dict) -> str:
    """Return a human-readable summary of the knowledge graph."""
    lines = [f"Programs: {len(kg['programs'])}"]
    for pid, info in kg['programs'].items():
        lines.append(
            f"  {pid}: {len(info['paragraphs'])} paragraphs, "
            f"{len(info['data_items'])} data items, "
            f"{len(info['business_rules'])} rules, "
            f"fan_in={info['fan_in']}, fan_out={info['fan_out']}"
        )
    lines.append(f"Edges: {len(kg['edges'])}")
    return "\n".join(lines)
