"""
Tests for the COBEvolve-specific Refactoring and Learning agent boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.learning_agent import record_module_outcome
from agents.refactoring_agent import refactor_program
from core.knowledge_base import KnowledgeBase
from utils.cobol_parser import parse_cobol_source


SAMPLE_COBOL = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. SIMPLE.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-VALUE PIC 9 VALUE 0.
       PROCEDURE DIVISION.
       MAIN-LOGIC.
           PERFORM USED-PARA.
           STOP RUN.
       USED-PARA.
           DISPLAY "OK".
       UNUSED-PARA.
           DISPLAY "NOT REACHED".
"""


def test_refactoring_agent_records_safe_opportunities():
    program = parse_cobol_source(SAMPLE_COBOL, "SIMPLE.cbl")

    result = refactor_program(program)

    assert result.status in {"ANALYZED", "NORMALIZED"}
    assert result.refactored_source.endswith("\n")
    assert any("UNUSED-PARA" in item for item in result.opportunities)


def test_learning_agent_records_cycle_in_knowledge_base(tmp_path):
    kb = KnowledgeBase(
        db_path=str(tmp_path / "kb.db"),
        chroma_path=str(tmp_path / "chroma"),
    )
    outcome = {
        "program_id": "SIMPLE",
        "translation_status": "RULE",
        "validation_status": "PASS",
        "repair_status": "ALREADY_PASSING",
        "pass_rate": 1.0,
    }

    snapshot = record_module_outcome(kb, outcome, agent_notes={"refactor": "analyzed"})

    assert snapshot.status == "RECORDED"
    assert "translation:rule" in snapshot.learned_patterns
    assert "validation:behaviour_preserved" in snapshot.learned_patterns
    with kb._connect() as conn:
        row = conn.execute(
            "SELECT notes FROM migration_log WHERE program_id=? AND module_path=?",
            ("SIMPLE", "LEARN"),
        ).fetchone()
    assert row is not None
    assert json.loads(row["notes"])["validation_status"] == "PASS"
