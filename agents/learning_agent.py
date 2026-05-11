
"""
agents/learning_agent.py -- Learning Agent.

COBEvolve role:
  Record outcomes from each modernisation pass so future passes can reuse
  successful translations, failure diagnoses, and agent decisions.

The existing KnowledgeBase remains the persistence layer. This agent gives that
knowledge step an explicit architecture boundary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.knowledge_base import KnowledgeBase, MigrationEvent

logger = logging.getLogger(__name__)


@dataclass
class LearningSnapshot:
    program_id: str
    status: str = "RECORDED"
    learned_patterns: list[str] = field(default_factory=list)
    kb_stats: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "program_id": self.program_id,
            "status": self.status,
            "learned_patterns": self.learned_patterns,
            "kb_stats": self.kb_stats,
            "generated_at": self.generated_at,
        }


def _patterns_from_outcome(outcome: dict[str, Any]) -> list[str]:
    patterns: list[str] = []
    translation_status = outcome.get("translation_status", "")
    validation_status = outcome.get("validation_status", "")
    repair_status = outcome.get("repair_status", "")

    if translation_status in {"RULE", "CACHE", "HYBRID", "LLM"}:
        patterns.append(f"translation:{translation_status.lower()}")
    if validation_status == "PASS":
        patterns.append("validation:behaviour_preserved")
    elif validation_status in {"PARTIAL", "FAIL"}:
        patterns.append(f"validation:{validation_status.lower()}")
    elif validation_status == "SKIPPED":
        patterns.append("validation:oracle_unavailable")
    if repair_status == "FIXED":
        patterns.append("repair:successful")
    elif repair_status == "ROLLBACK":
        patterns.append("repair:rollback")

    return patterns


def record_module_outcome(
    kb: KnowledgeBase,
    outcome: dict[str, Any],
    agent_notes: dict[str, Any] | None = None,
) -> LearningSnapshot:
    """Persist a compact learning event for one module cycle."""
    program_id = outcome.get("program_id", "UNKNOWN")
    patterns = _patterns_from_outcome(outcome)
    if agent_notes:
        for key, value in agent_notes.items():
            if value:
                patterns.append(f"{key}:{value}")

    notes = {
        "patterns": patterns,
        "validation_status": outcome.get("validation_status"),
        "repair_status": outcome.get("repair_status"),
        "pass_rate": outcome.get("pass_rate"),
    }
    kb.log_event(
        MigrationEvent(
            program_id=program_id,
            module_path="LEARN",
            status="RECORDED",
            notes=json.dumps(notes, sort_keys=True),
        )
    )
    snapshot = LearningSnapshot(
        program_id=program_id,
        learned_patterns=patterns,
        kb_stats=kb.stats(),
    )
    logger.info("[%s] LearningAgent: recorded %d pattern(s)", program_id, len(patterns))
    return snapshot


def summarise_learning(snapshot: LearningSnapshot) -> str:
    return f"LearningAgent {snapshot.status} ({len(snapshot.learned_patterns)} patterns)"


def summarise_learning_across_passes(kb: KnowledgeBase) -> dict:
    """Produce a cross-pass learning summary for paper §4 evidence.
    Call this at the end of each pipeline run to document self-evolution."""
    summary = kb.get_learning_summary()
    logger.info(
        "Cross-pass learning summary: %d translations cached, "
        "%d cache hits, %d semantic embeddings",
        summary['total_translations_cached'],
        summary['cache_hits_across_passes'],
        summary['semantic_index_size'],
    )
    return summary


def compare_passes(kb: KnowledgeBase, stats_before: dict, stats_after: dict) -> dict:
    """Compare KB state before and after a pass to quantify learning growth."""
    return {
        "new_translations": stats_after.get('translations', 0) - stats_before.get('translations', 0),
        "new_embeddings": stats_after.get('chroma_embeddings', 0) - stats_before.get('chroma_embeddings', 0),
        "new_events": stats_after.get('events', 0) - stats_before.get('events', 0),
        "cache_hits_this_pass": stats_after.get('cache_hits_across_passes', 0) - stats_before.get('cache_hits_across_passes', 0),
    }
