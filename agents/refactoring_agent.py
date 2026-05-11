"""
agents/refactoring_agent.py -- Refactoring Agent.

COBEvolve role:
  Improve COBOL structure before translation while preserving behaviour.

This first implementation is deliberately conservative. It performs safe source
normalisation and records refactoring opportunities, but it does not delete or
rewrite executable COBOL paragraphs by default. That keeps the benchmark path
stable while making the RefactoringAgent an explicit part of the MAPE-K loop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.knowledge_base import KnowledgeBase, MigrationEvent
from utils.cobol_parser import COBOLProgram

logger = logging.getLogger(__name__)


@dataclass
class RefactoringResult:
    program_id: str
    status: str = "ANALYZED"  # ANALYZED | NORMALIZED | SKIPPED | FAILED
    changed: bool = False
    refactored_source: str = ""
    opportunities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "program_id": self.program_id,
            "status": self.status,
            "changed": self.changed,
            "opportunities": self.opportunities,
            "warnings": self.warnings,
            "generated_at": self.generated_at,
        }


def _normalise_source(source: str) -> str:
    """Remove trailing whitespace, collapse long blank runs, ensure final newline."""
    lines = [line.rstrip() for line in source.splitlines()]
    normalised = "\n".join(lines)
    normalised = re.sub(r"\n{4,}", "\n\n\n", normalised)
    return normalised.rstrip() + "\n" if normalised.strip() else source


def _reachable_paragraphs(program: COBOLProgram) -> set[str]:
    """Compute paragraphs reachable from the first paragraph via PERFORM edges."""
    if not program.paragraphs:
        return set()

    perform_map = {paragraph.name: set(paragraph.performs) for paragraph in program.paragraphs}
    known = set(perform_map)
    reachable = {program.paragraphs[0].name}
    stack = [program.paragraphs[0].name]

    while stack:
        current = stack.pop()
        for target in perform_map.get(current, set()):
            if target in known and target not in reachable:
                reachable.add(target)
                stack.append(target)

    return reachable


def _find_opportunities(program: COBOLProgram) -> list[str]:
    """Return human-readable refactoring opportunities."""
    opportunities = []
    reachable = _reachable_paragraphs(program)
    all_paras = {p.name for p in program.paragraphs}
    dead = all_paras - reachable

    if dead:
        for p in sorted(dead):
            opportunities.append(f"Dead paragraph (unreachable): {p}")

    # Detect duplicated paragraph names (defensive)
    seen: set[str] = set()
    for para in program.paragraphs:
        if para.name in seen:
            opportunities.append(f"Duplicate paragraph name: {para.name}")
        seen.add(para.name)

    # Detect very long paragraphs (candidate for decomposition)
    for para in program.paragraphs:
        lines = para.raw_body.strip().splitlines()
        if len(lines) > 50:
            opportunities.append(
                f"Long paragraph ({len(lines)} lines, candidate for decomposition): {para.name}"
            )

    return opportunities


def remove_dead_paragraphs(
    program: COBOLProgram,
    kb: KnowledgeBase | None = None,
) -> RefactoringResult:
    """
    Detect and record unreachable paragraphs.
    Conservative: records opportunities without removing, preserving behaviour.
    This is safe for the MAPE-K pipeline and satisfies the paper claim
    'identifies refactoring opportunities recorded in the KB for human review'.
    """
    opportunities = _find_opportunities(program)
    dead = [o for o in opportunities if o.startswith("Dead paragraph")]

    result = RefactoringResult(
        program_id=program.program_id,
        status="ANALYZED" if not dead else "NORMALIZED",
        changed=False,
        refactored_source=program.raw_source,
        opportunities=opportunities,
    )

    if kb is not None and opportunities:
        kb.log_event(MigrationEvent(
            program_id=program.program_id,
            module_path="REFACTOR",
            status="OPPORTUNITIES_RECORDED",
            notes=f"{len(dead)} dead paragraphs, {len(opportunities)} total opportunities",
        ))

    logger.info(
        "[%s] RefactoringAgent: %d opportunities found (%d dead paragraphs)",
        program.program_id, len(opportunities), len(dead)
    )
    return result


def refactor_program(
    program: COBOLProgram,
    kb: KnowledgeBase | None = None,
) -> RefactoringResult:
    """
    Analyze and safely normalise a COBOLProgram before translation.

    The returned source is semantically equivalent to the input source. Structural
    opportunities are recorded for human review and future COBEvolve learning.
    """
    source = program.raw_source or ""
    if not source.strip():
        return RefactoringResult(
            program_id=program.program_id,
            status="SKIPPED",
            refactored_source=source,
            warnings=["empty source"],
        )

    refactored = _normalise_source(source)
    changed = refactored != source

    # Use remove_dead_paragraphs to detect opportunities and log to KB
    dead_result = remove_dead_paragraphs(program, kb=kb)
    opportunities = dead_result.opportunities
    status = "NORMALIZED" if changed else dead_result.status

    result = RefactoringResult(
        program_id=program.program_id,
        status=status,
        changed=changed,
        refactored_source=refactored,
        opportunities=opportunities,
    )

    logger.info("[%s] RefactoringAgent: %s", program.program_id, status)
    return result


def summarise_refactoring(result: RefactoringResult) -> str:
    suffix = ""
    if result.opportunities:
        suffix = f" ({len(result.opportunities)} opportunities)"
    return f"RefactoringAgent {result.status}{suffix}"
