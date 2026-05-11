"""
Regression tests for issues from BugReport_and_Fixes.md that remain relevant
to the current COBEvolve implementation.
"""

from __future__ import annotations

from agents.validation_agent import (
    ValidationResult,
    format_validation_report,
    summarise_validation,
)
from core.orchestrator import ModuleOutcome, PipelineReport, _tally_report_counts
from utils.cobol_parser import parse_cobol_source


def test_validation_report_function_exists_for_translate_cli():
    vr = ValidationResult(
        program_id="SIMPLE",
        status="FAIL",
        pass_rate=0.0,
        passing_cases=0,
        failing_cases=1,
        error_cases=0,
        test_results=[{
            "test_id": "tc_001",
            "status": "FAIL",
            "expected": "A",
            "actual": "B",
            "diff": "Line 1 differs",
        }],
    )

    assert "SIMPLE: FAIL" in summarise_validation(vr)
    report = format_validation_report(vr)
    assert "Validation Report: SIMPLE" in report
    assert "Line 1 differs" in report


def test_parser_handles_missing_data_division_header():
    source = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. NODATAHDR.
       WORKING-STORAGE SECTION.
       01 WS-AMOUNT PIC 9(5) VALUE 10.
       PROCEDURE DIVISION.
       MAIN-LOGIC.
           DISPLAY WS-AMOUNT.
           STOP RUN.
"""

    program = parse_cobol_source(source, "NODATAHDR.cbl")

    assert program.program_id == "NODATAHDR"
    assert [item.name for item in program.working_storage] == ["WS-AMOUNT"]


def test_pipeline_report_counts_fail_skipped_as_failed():
    report = PipelineReport()
    outcomes = [
        ModuleOutcome(
            program_id="BROKEN",
            cycle=1,
            refactor_status="ANALYZED",
            translation_status="RULE",
            validation_status="FAIL",
            repair_status="SKIPPED",
        )
    ]

    _tally_report_counts(report, outcomes)

    assert report.total_modules == 1
    assert report.failed == 1
    assert report.to_dict()["failed"] == 1
    assert "Failed            : 1" in report.summary_text()
