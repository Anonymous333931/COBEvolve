"""
core/orchestrator.py -- MAPE-K Loop Controller.

Coordinates the COBEvolve MAPE-K pipeline:

  Monitor  → Comprehension Agent  (scan repo, build knowledge graph)
  Analyse  → MAPE-K controller    (RVF scoring, migration plan)
  Plan     → Refactoring Agent    (safe COBOL restructuring opportunities)
  Execute  → Translation Agent    (translate module)
           → TestGeneration Agent (oracle generation)
           → Validation Agent     (behavioural equivalence)
  Knowledge→ Learning Agent       (record outcomes in KnowledgeBase)

Self-repair remains an internal remediation service used when validation fails;
it is not one of the six paper-facing COBEvolve agents.

REPO-WISE: The orchestrator handles entire repository trees.
Pass a directory path (not a single file) to run() for full repo analysis.

IMPORTANT: comprehend() now returns (kg_dict, graph) tuple — Bug #5 fix.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from crewai import Agent, Crew, Process, Task
from langchain_ollama import OllamaLLM

from config import config
from core.knowledge_base import KnowledgeBase, MigrationEvent
from agents.comprehension_agent import (
    comprehend, comprehend_multiple_repos, summarise_knowledge_graph,
)
from agents.planning_agent import (
    plan, MigrationPlan, ScoredModule, summarise_plan,
)
from agents.translation_agent import (
    translate_module, TranslationResult, summarise_results,
)
from agents.test_generation_agent import generate_oracle, BehavioralOracle
from agents.validation_agent import (
    validate, ValidationResult, summarise_validation,
)
from agents.self_repair_agent import repair, RepairResult, summarise_repair
from agents.refactoring_agent import (
    RefactoringResult, refactor_program, summarise_refactoring,
)
from agents.learning_agent import (
    LearningSnapshot, record_module_outcome, summarise_learning,
)
from utils.cobol_parser import parse_cobol_source

logger = logging.getLogger(__name__)


_ORACLE_SKIP_MARKERS = (
    "executable program requested but procedure/entry has using clause",
    "not a standalone executable program",
    "not suitable for non-interactive oracle execution",
)


# ════════════════════════════════════════════════════════════════
# Data models
# ════════════════════════════════════════════════════════════════

@dataclass
class ModuleOutcome:
    """Outcome record for a single module processed in one pipeline cycle."""
    program_id: str
    cycle: int
    refactor_status: str          # ANALYZED / NORMALIZED / SKIPPED / FAILED
    translation_status: str        # RULE / LLM / HYBRID / CACHED / FAILED
    validation_status: str         # PASS / PARTIAL / FAIL / SKIPPED
    repair_status: str             # FIXED / ROLLBACK / ALREADY_PASSING / SKIPPED
    learning_status: str = "PENDING"
    pass_rate: float = 0.0
    attempts: int = 1
    elapsed_seconds: float = 0.0
    output_filepath: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "program_id":         self.program_id,
            "cycle":              self.cycle,
            "refactor_status":    self.refactor_status,
            "translation_status": self.translation_status,
            "validation_status":  self.validation_status,
            "repair_status":      self.repair_status,
            "learning_status":    self.learning_status,
            "pass_rate":          round(self.pass_rate, 4),
            "attempts":           self.attempts,
            "elapsed_seconds":    round(self.elapsed_seconds, 2),
            "output_filepath":    self.output_filepath,
            "notes":              self.notes,
        }


@dataclass
class PipelineReport:
    """Summary written to modernised_output/pipeline_report.json."""
    total_modules: int = 0
    auto_modernised: int = 0
    partially_modernised: int = 0
    rolled_back: int = 0
    skipped: int = 0
    failed: int = 0
    total_cycles: int = 0
    elapsed_seconds: float = 0.0
    outcomes: list[ModuleOutcome] = field(default_factory=list)
    kb_stats: dict = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def automation_rate(self) -> float:
        if self.total_modules == 0:
            return 0.0
        return round(self.auto_modernised / self.total_modules, 4)

    def to_dict(self) -> dict:
        return {
            "total_modules":       self.total_modules,
            "auto_modernised":     self.auto_modernised,
            "partially_modernised": self.partially_modernised,
            "rolled_back":         self.rolled_back,
            "skipped":             self.skipped,
            "failed":              self.failed,
            "automation_rate":     self.automation_rate,
            "total_cycles":        self.total_cycles,
            "elapsed_seconds":     round(self.elapsed_seconds, 2),
            "kb_stats":            self.kb_stats,
            "generated_at":        self.generated_at,
            "outcomes":            [o.to_dict() for o in self.outcomes],
        }

    def summary_text(self) -> str:
        bar = "=" * 60
        return (
            f"\n{bar}\n"
            f"Pipeline Complete\n"
            f"  Total modules     : {self.total_modules}\n"
            f"  Auto-modernised   : {self.auto_modernised}\n"
            f"  Partial           : {self.partially_modernised}\n"
            f"  Rolled back       : {self.rolled_back}\n"
            f"  Failed            : {self.failed}\n"
            f"  Skipped           : {self.skipped}\n"
            f"  Automation rate   : {self.automation_rate:.1%}\n"
            f"  Total time        : {self.elapsed_seconds:.1f}s\n"
            f"  KB translations   : {self.kb_stats.get('translations', 0)}\n"
            f"  KB patterns stored: {self.kb_stats.get('chroma_embeddings', 0)}\n"
            f"{bar}"
        )


def _tally_report_counts(report: PipelineReport, outcomes: list[ModuleOutcome]) -> None:
    """Populate report outcome buckets, including explicit failed modules."""
    report.total_modules = len(outcomes)
    report.total_cycles = len(outcomes)
    for outcome in outcomes:
        if outcome.validation_status == "PASS":
            report.auto_modernised += 1
        elif outcome.validation_status == "PARTIAL":
            report.partially_modernised += 1
        elif outcome.repair_status == "ROLLBACK":
            report.rolled_back += 1
        elif outcome.validation_status == "SKIPPED":
            report.skipped += 1
        else:
            report.failed += 1


# ════════════════════════════════════════════════════════════════
# CrewAI agent + task factories
# ════════════════════════════════════════════════════════════════

def _make_llm() -> OllamaLLM:
    """Create an Ollama-backed LLM for CrewAI agents."""
    return OllamaLLM(
        model=config.MODEL_ANALYSIS,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
    )


def _make_crew_agents(llm: OllamaLLM) -> dict[str, Agent]:
    """Build all CrewAI agent objects (one per pipeline role)."""
    return {
        "comprehension": Agent(
            role="COBOL Comprehension Specialist",
            goal=(
                "Parse the COBOL codebase, build a dependency graph, "
                "and extract business rules from every module."
            ),
            backstory=(
                "You are a senior mainframe engineer with 30 years of COBOL "
                "experience. You understand COBOL data divisions, paragraph "
                "flow, and the implicit business rules baked into legacy code."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        ),
        "refactoring": Agent(
            role="COBOL Refactoring Specialist",
            goal=(
                "Improve COBOL structure before translation while preserving "
                "behaviour and recording safe refactoring opportunities."
            ),
            backstory=(
                "You specialise in legacy COBOL maintainability. You identify "
                "dead paragraphs, duplicated control-flow patterns, and layout "
                "sensitive constructs that require careful treatment."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        ),
        "translation": Agent(
            role="COBOL-to-Python Translator",
            goal=(
                "Translate COBOL modules to correct, idiomatic Python "
                "using rule-based patterns supplemented by LLM for gaps."
            ),
            backstory=(
                "You are an expert in COB2PY-style COBOL translation. "
                "You understand COBOL arithmetic, string handling, and "
                "data division semantics and can map them to Python."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        ),
        "test_generation": Agent(
            role="COBOL Test Generation Specialist",
            goal=(
                "Generate behavioural oracle tests from original COBOL "
                "components so modernised code can be checked for equivalence."
            ),
            backstory=(
                "You specialise in regression testing for legacy migration. "
                "You create executable oracle cases from COBOL behaviour and "
                "identify when programs are not runnable in isolation."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        ),
        "validation": Agent(
            role="Behavioural Equivalence Validator",
            goal=(
                "Verify that translated Python produces identical outputs "
                "to the original COBOL for all test cases."
            ),
            backstory=(
                "You are a QA engineer specialising in COBOL-to-Python "
                "migration testing. You use GnuCOBOL oracle results to "
                "validate behavioural equivalence with numeric tolerance."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        ),
        "learning": Agent(
            role="Modernisation Learning Engineer",
            goal=(
                "Record each pass into the knowledge base so later runs can "
                "reuse successful patterns and avoid repeated failures."
            ),
            backstory=(
                "You operate the Knowledge component of the MAPE-K loop. You "
                "turn agent decisions and validation outcomes into reusable "
                "evolution knowledge."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        ),
    }


# ════════════════════════════════════════════════════════════════
# Per-module execution cycle
# ════════════════════════════════════════════════════════════════

def _run_module_cycle(
    module: ScoredModule,
    kg: dict,
    kb: KnowledgeBase,
    cycle: int,
    output_dir: str,
    use_llm: bool = True,
) -> ModuleOutcome:
    """
    Run the full PLAN→EXECUTE cycle for one COBOL module:
      Translate → TestGen → Validate → (Repair if needed)
    """
    pid = module.program_id
    start = time.time()
    outcome = ModuleOutcome(
        program_id=pid,
        cycle=cycle,
        refactor_status="SKIPPED",
        translation_status="FAILED",
        validation_status="SKIPPED",
        repair_status="SKIPPED",
    )

    def _learn_and_return(
        refactor_result: RefactoringResult | None = None,
    ) -> ModuleOutcome:
        agent_notes = {}
        if refactor_result is not None:
            agent_notes["refactor"] = refactor_result.status.lower()
            if refactor_result.opportunities:
                agent_notes["refactor_opportunities"] = len(refactor_result.opportunities)
        try:
            snapshot: LearningSnapshot = record_module_outcome(
                kb, outcome.to_dict(), agent_notes=agent_notes
            )
            outcome.learning_status = snapshot.status
            logger.info("[%s] %s", pid, summarise_learning(snapshot))
        except Exception as exc:
            outcome.learning_status = "FAILED"
            note = f"Learning error: {exc}"
            outcome.notes = f"{outcome.notes}; {note}" if outcome.notes else note
            logger.warning("[%s] %s", pid, note)
        return outcome

    # Reconstruct COBOLProgram from raw_source in kg
    prog_info = kg["programs"].get(pid, {})
    raw_source = prog_info.get("raw_source", "")
    filepath = prog_info.get("filepath", "")
    if not raw_source:
        logger.warning("[%s] No raw_source in knowledge graph — skipping", pid)
        outcome.notes = "No raw_source in knowledge graph"
        return _learn_and_return()

    program = parse_cobol_source(raw_source, filepath=filepath)

    # ── KB-informed strategy (self-evolution evidence) ────────────────────
    kb_strategy = kb.kb_informed_strategy(pid)
    logger.info("[%s] KB recommends strategy: %s", pid, kb_strategy)
    kb.log_event(MigrationEvent(pid, "STRATEGY", kb_strategy,
                                details=f"kb_recommended={kb_strategy}"))

    # ── Refactoring ───────────────────────────────────────────────────────
    refactor_result: RefactoringResult | None = None
    try:
        refactor_result = refactor_program(program, kb=kb)
        outcome.refactor_status = refactor_result.status
        logger.info("[%s] %s", pid, summarise_refactoring(refactor_result))
        if refactor_result.changed:
            raw_source = refactor_result.refactored_source
            program = parse_cobol_source(raw_source, filepath=filepath)
    except Exception as exc:
        outcome.refactor_status = "FAILED"
        outcome.notes = f"Refactoring error: {exc}"
        logger.warning("[%s] Refactoring failed; continuing with original source: %s", pid, exc)

    kb.log_event(MigrationEvent(pid, "TRANSLATE", "START", details=filepath))

    # ── Translation ────────────────────────────────────────────────────────
    try:
        tr: TranslationResult = translate_module(
            module, program, kb,
            use_llm=use_llm,
            output_dir=output_dir,
        )
        outcome.translation_status = tr.method.upper()
        outcome.output_filepath = tr.output_filepath
        logger.info("[%s] Translated (method=%s, coverage=%.0f%%)",
                    pid, tr.method, tr.rule_coverage * 100)
    except Exception as exc:
        logger.error("[%s] Translation crashed: %s", pid, exc)
        outcome.notes = f"Translation error: {exc}"
        outcome.elapsed_seconds = time.time() - start
        return _learn_and_return(refactor_result)

    kb.log_event(MigrationEvent(pid, "TRANSLATE", "DONE",
                                details=f"method={tr.method}"))

    # ── Test generation ────────────────────────────────────────────────────
    try:
        oracle: BehavioralOracle = generate_oracle(program, output_dir=output_dir)
    except Exception as exc:
        logger.warning("[%s] Oracle generation failed: %s — skipping validation", pid, exc)
        oracle = BehavioralOracle(
            program_id=pid,
            cobol_filepath=filepath,
            compile_success=False,
            compile_error=str(exc),
        )

    # ── Validation ─────────────────────────────────────────────────────────
    try:
        vr: ValidationResult = validate(tr, oracle, cobol_source=raw_source)
        outcome.validation_status = vr.status
        outcome.pass_rate = vr.pass_rate
        logger.info("[%s] %s", pid, summarise_validation(vr))
    except Exception as exc:
        logger.error("[%s] Validation crashed: %s", pid, exc)
        outcome.notes = f"Validation error: {exc}"
        outcome.elapsed_seconds = time.time() - start
        return _learn_and_return(refactor_result)

    kb.log_event(MigrationEvent(pid, "VALIDATE", vr.status,
                                details=f"pass_rate={vr.pass_rate:.2f}"))

    # ── Self-repair (if needed) ────────────────────────────────────────────
    if not oracle.compile_success or not oracle.test_cases:
        outcome.repair_status = "SKIPPED"
        compile_error = oracle.compile_error or "No oracle test cases available"
        lowered_error = compile_error.lower()
        if any(marker in lowered_error for marker in _ORACLE_SKIP_MARKERS):
            outcome.validation_status = "SKIPPED"
            if "not suitable for non-interactive oracle execution" in lowered_error:
                outcome.notes = "Interactive screen/menu program; oracle skipped"
            else:
                outcome.notes = "Non-standalone subprogram; oracle skipped"
        else:
            outcome.notes = compile_error
    elif not use_llm:
        outcome.repair_status = "SKIPPED"
        outcome.notes = "LLM disabled; self-repair skipped"
    elif vr.status in ("FAIL", "PARTIAL"):
        logger.info("[%s] Validation %s — invoking remediation backend", pid, vr.status)
        try:
            rr: RepairResult = repair(tr, vr, oracle, kb)
            outcome.repair_status = rr.status
            outcome.attempts = rr.attempts
            logger.info("[%s] %s", pid, summarise_repair(rr))
            if rr.status == "FIXED":
                outcome.validation_status = "PASS"
                outcome.pass_rate = rr.final_pass_rate
                outcome.output_filepath = rr.output_filepath
            elif rr.status == "ROLLBACK":
                outcome.output_filepath = None   # Bug #17 sentinel
        except Exception as exc:
            logger.error("[%s] Self-repair crashed: %s", pid, exc)
            outcome.repair_status = "CRASHED"
            outcome.notes = f"Repair error: {exc}"

        kb.log_event(MigrationEvent(pid, "REPAIR", outcome.repair_status))
    else:
        outcome.repair_status = "ALREADY_PASSING"

    outcome.elapsed_seconds = time.time() - start
    logger.info(
        "[%s] Cycle done in %.1fs: refactor=%s translate=%s validate=%s repair=%s",
        pid, outcome.elapsed_seconds,
        outcome.refactor_status, outcome.translation_status, outcome.validation_status,
        outcome.repair_status,
    )
    return _learn_and_return(refactor_result)


# ════════════════════════════════════════════════════════════════
# Main orchestrator class
# ════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    MAPE-K Orchestrator for the Self-Evolving COBOL Pipeline.

    Usage
    -----
    orch = Orchestrator()
    report = orch.run("./samples/bank_system")
    print(report.summary_text())

    # Multiple repos at once (repo-wise):
    report = orch.run_repos(["./samples/bank_system", "./samples/payroll_system"])
    """

    def __init__(
        self,
        db_path: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        config.setup()
        self.kb = KnowledgeBase(db_path=db_path)
        self.output_dir = output_dir or config.OUTPUT_DIR
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._llm: Optional[OllamaLLM] = None
        self._crew_agents: Optional[dict] = None

    def _get_crew_agents(self) -> dict[str, Agent]:
        if self._crew_agents is None:
            self._llm = _make_llm()
            self._crew_agents = _make_crew_agents(self._llm)
        return self._crew_agents

    # ── MONITOR ───────────────────────────────────────────────────────────

    def monitor(
        self,
        source: str,
        extract_rules: bool = True,
    ) -> tuple[dict, object]:
        """MONITOR phase: comprehend the codebase."""
        logger.info("=== MONITOR: Comprehension Agent ===")
        # Bug #5 fix: comprehend() returns (kg, graph) tuple
        kg, graph = comprehend(source, extract_rules=extract_rules,
                               repo_root=source if Path(source).is_dir() else "")
        logger.info("\n%s", summarise_knowledge_graph(kg))
        return kg, graph

    # ── ANALYZE ───────────────────────────────────────────────────────────

    def analyze(self, kg: dict, graph) -> MigrationPlan:
        """ANALYZE phase: score and order modules."""
        logger.info("=== ANALYZE: MAPE-K Controller Planning ===")
        # Pass graph explicitly (not embedded in kg — Bug #5 fix)
        kg_with_graph = dict(kg)
        kg_with_graph["_graph"] = graph
        mp = plan(kg_with_graph, output_path=str(
            Path(self.output_dir) / "migration_plan.json"
        ))
        logger.info("\n%s", summarise_plan(mp))
        return mp

    # ── EXECUTE (all modules) ─────────────────────────────────────────────

    def execute(
        self,
        migration_plan: MigrationPlan,
        kg: dict,
        max_modules: Optional[int] = None,
        use_llm: bool = True,
    ) -> list[ModuleOutcome]:
        """EXECUTE phase: translate, validate, repair for each module."""
        logger.info("=== EXECUTE: Processing %d module(s) ===",
                    len(migration_plan.modules))
        outcomes: list[ModuleOutcome] = []
        modules = migration_plan.modules
        if max_modules:
            modules = modules[:max_modules]

        for i, module in enumerate(modules, 1):
            logger.info(
                "--- Module %d/%d: %s (priority=%d) ---",
                i, len(modules), module.program_id, module.priority,
            )
            outcome = _run_module_cycle(
                module=module,
                kg=kg,
                kb=self.kb,
                cycle=i,
                output_dir=self.output_dir,
                use_llm=use_llm,
            )
            outcomes.append(outcome)

        return outcomes

    # ── Full pipeline run ─────────────────────────────────────────────────

    def run(
        self,
        source: str,
        extract_rules: bool = True,
        max_modules: Optional[int] = None,
        use_crewai: bool = False,
        use_llm: bool = True,
    ) -> PipelineReport:
        """
        Run the full MAPE-K pipeline over a COBOL codebase.

        Parameters
        ----------
        source        : path to a .cbl file OR a directory (repo-wise)
        extract_rules : call Ollama for business rule extraction
        max_modules   : limit how many modules to process (None = all)
        use_crewai    : if True, wire agents through CrewAI Crew object

        Returns
        -------
        PipelineReport with full outcome details
        """
        config.setup()
        # Mark pass start for multi-pass tracking
        pass_num = self.kb.stats().get('events', 0) // max(1, 1) + 1
        self.kb.mark_pass_start(pass_num, [source] if isinstance(source, str) else list(source))
        pipeline_start = time.time()
        report = PipelineReport()

        # ── M — Monitor ───────────────────────────────────────────────────
        kg, graph = self.monitor(source, extract_rules=extract_rules)

        # ── A — Analyze ───────────────────────────────────────────────────
        mp = self.analyze(kg, graph)

        # ── P+E — Execute ─────────────────────────────────────────────────
        if use_crewai:
            outcomes = self._run_crewai(mp, kg)
        else:
            outcomes = self.execute(mp, kg, max_modules=max_modules, use_llm=use_llm)

        # ── K — Knowledge (tally outcomes) ───────────────────────────────
        _tally_report_counts(report, outcomes)
        report.outcomes = outcomes
        report.elapsed_seconds = time.time() - pipeline_start
        report.kb_stats = self.kb.stats()

        # Add learning summary to report
        learning_summary = self.kb.get_learning_summary()
        report.kb_stats = {**report.kb_stats, **learning_summary}
        logger.info("Learning summary: %s", learning_summary)

        # Write report JSON
        report_path = Path(self.output_dir) / "pipeline_report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        logger.info("Report written to %s", report_path)
        print(report.summary_text())
        return report

    def run_repos(
        self,
        repo_dirs: list[str],
        extract_rules: bool = True,
        max_modules: Optional[int] = None,
        use_llm: bool = True,
    ) -> PipelineReport:
        """
        REPO-WISE: Run the pipeline over multiple repositories together.
        All repos share one knowledge graph and dependency graph.
        """
        config.setup()
        pass_num = self.kb.stats().get('events', 0) + 1
        self.kb.mark_pass_start(pass_num, repo_dirs)
        pipeline_start = time.time()
        report = PipelineReport()

        logger.info("=== MULTI-REPO MONITOR ===")
        # Bug #5 fix: returns tuple
        kg, graph = comprehend_multiple_repos(repo_dirs, extract_rules=extract_rules)
        logger.info("\n%s", summarise_knowledge_graph(kg))

        mp = self.analyze(kg, graph)
        outcomes = self.execute(mp, kg, max_modules=max_modules, use_llm=use_llm)

        _tally_report_counts(report, outcomes)
        report.outcomes = outcomes
        report.elapsed_seconds = time.time() - pipeline_start
        report.kb_stats = self.kb.stats()

        # Add learning summary to report
        learning_summary = self.kb.get_learning_summary()
        report.kb_stats = {**report.kb_stats, **learning_summary}
        logger.info("Learning summary: %s", learning_summary)

        report_path = Path(self.output_dir) / "pipeline_report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        print(report.summary_text())
        return report

    # ── CrewAI wiring (optional) ──────────────────────────────────────────

    def _run_crewai(
        self,
        mp: MigrationPlan,
        kg: dict,
    ) -> list[ModuleOutcome]:
        """Run the pipeline through CrewAI for agent observability."""
        agents = self._get_crew_agents()
        outcomes: list[ModuleOutcome] = []

        for module in mp.modules:
            pid = module.program_id

            tasks = [
                Task(
                    description=f"Inspect COBOL module {pid} for safe refactoring opportunities.",
                    expected_output="Refactoring status and preservation notes.",
                    agent=agents["refactoring"],
                ),
                Task(
                    description=f"Translate COBOL module {pid} to Python.",
                    expected_output="Python source code for the module.",
                    agent=agents["translation"],
                ),
                Task(
                    description=f"Generate oracle tests for COBOL module {pid}.",
                    expected_output="Behavioural oracle status and generated test summary.",
                    agent=agents["test_generation"],
                ),
                Task(
                    description=f"Validate the translated Python for {pid}.",
                    expected_output="Validation result: PASS, PARTIAL, or FAIL.",
                    agent=agents["validation"],
                ),
                Task(
                    description=f"Record the modernisation outcome for {pid}.",
                    expected_output="Learning event recorded in the knowledge base.",
                    agent=agents["learning"],
                ),
            ]

            crew = Crew(
                agents=list(agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
            )

            try:
                crew.kickoff()
            except Exception as exc:
                logger.warning("[%s] CrewAI run failed: %s — falling back", pid, exc)

            # Always run the actual agents directly for correctness
            outcome = _run_module_cycle(
                module=module, kg=kg, kb=self.kb,
                cycle=len(outcomes) + 1,
                output_dir=self.output_dir,
            )
            outcomes.append(outcome)

        return outcomes


# Backward-compatible alias used by older Step 7 documentation.
MAPEKOrchestrator = Orchestrator


def run_crewai_pipeline(
    source: str | list[str],
    db_path: str | None = None,
    output_dir: str | None = None,
    extract_rules: bool = True,
    max_modules: Optional[int] = None,
    use_llm: bool = True,
) -> PipelineReport:
    """
    Compatibility wrapper for guide examples that refer to
    ``run_crewai_pipeline()``.

    Single-source runs use the CrewAI execution path. Multi-repo runs reuse
    the existing repo-wise runner.
    """
    orch = Orchestrator(db_path=db_path, output_dir=output_dir)
    if isinstance(source, (list, tuple)):
        sources = list(source)
        if len(sources) == 1:
            return orch.run(
                sources[0],
                extract_rules=extract_rules,
                max_modules=max_modules,
                use_crewai=True,
                use_llm=use_llm,
            )
        return orch.run_repos(
            sources,
            extract_rules=extract_rules,
            max_modules=max_modules,
            use_llm=use_llm,
        )
    return orch.run(
        source,
        extract_rules=extract_rules,
        max_modules=max_modules,
        use_crewai=True,
        use_llm=use_llm,
    )
