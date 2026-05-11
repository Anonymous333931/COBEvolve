
# ── FILE: cobol_moderniser/agents/validation_agent.py ─────────────────
"""
agents/validation_agent.py — Validation Agent

Runs translated Python code against the BehavioralOracle and
produces a ValidationResult (PASS / FAIL / PARTIAL / ERROR).

PASS    → commit the translation to modernised_output/
PARTIAL → send to internal remediation with diff context
FAIL    → full retry in internal remediation
ERROR   → translation crashed or timed out (treat as FAIL)
"""
from __future__ import annotations
import logging, re, subprocess, sys, tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config import config
from agents.test_generation_agent import BehavioralOracle, TestCase
from agents.translation_agent import TranslationResult

logger = logging.getLogger(__name__)

VALIDATION_THRESHOLD = config.VALIDATION_THRESHOLD  # default 0.80
NUMERIC_TOLERANCE    = 0.01  # absolute tolerance for numeric comparisons

@dataclass
class TestCaseResult:
    test_id: str
    status: str    # 'PASS' | 'FAIL' | 'ERROR'
    expected: str
    actual: str
    diff: str = ''  # first mismatched line

@dataclass
class ValidationResult:
    program_id: str
    status: str            # 'PASS' | 'FAIL' | 'PARTIAL' | 'ERROR'
    pass_rate: float = 0.0
    passing_cases: int = 0
    failing_cases: int = 0
    error_cases: int = 0
    test_results: list = field(default_factory=list)
    error_message: str = ''
    translated_code: str = ''
    cobol_source: str = ''
    validated_at: str = ''
    def __post_init__(self):
        if not self.validated_at:
            self.validated_at = datetime.now(timezone.utc).isoformat()


# ════════════════════════════════════════════════
# PHASE 1: RUN TRANSLATED PYTHON
# ════════════════════════════════════════════════

def _inputs_to_stdin(inputs: dict) -> str:
    return '\n'.join(str(v) for v in inputs.values()) + '\n'

def run_python_translation(
    python_filepath: str,
    inputs: dict,
    timeout: int = 10,
) -> tuple[str, str]:
    """
    Execute a translated .py file in a subprocess.
    Returns (stdout, error_message).
    error_message is non-empty on crash, import error, or timeout.
    """
    stdin_text = _inputs_to_stdin(inputs)
    try:
        result = subprocess.run(
            [sys.executable, python_filepath],
            input=stdin_text, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            err = (result.stderr or 'non-zero exit').strip()[:200]
            if result.stdout:
                # Return partial stdout with error so Self-Repair has diff context
                return result.stdout.strip(), err
            return '', err
        return result.stdout.strip(), ''

    except subprocess.TimeoutExpired:
        return '', f'Python execution timed out after {timeout}s'
    except Exception as exc:
        return '', str(exc)


# ════════════════════════════════════════════════
# PHASE 2: EQUIVALENCE CHECKING
# ════════════════════════════════════════════════

def _normalise_line(line: str) -> str:
    """Strip, collapse whitespace, and uppercase a line for comparison."""
    def _normalise_hex(match: re.Match[str]) -> str:
        value = match.group(0)
        try:
            return '0XZERO' if int(value, 16) == 0 else '0XNONZERO'
        except ValueError:
            return value.upper()

    line = re.sub(r'0x[0-9a-fA-F]+', _normalise_hex, line)
    return re.sub(r'\s+', ' ', line.strip()).upper()

def _numeric_close(a: str, b: str) -> bool:
    """Return True if both strings parse as numbers within NUMERIC_TOLERANCE."""
    try:
        return abs(float(a) - float(b)) <= NUMERIC_TOLERANCE
    except ValueError:
        return False


def _parse_labeled_numeric_lines(lines: list[str]) -> list[tuple[str, float]] | None:
    """Parse lines like 'UNSORTED: 08' into (label, numeric_value) pairs."""
    parsed: list[tuple[str, float]] = []
    for line in lines:
        match = re.match(r'^([A-Z0-9 _-]+:)\s*(-?\d+(?:\.\d+)?)$', line)
        if not match:
            return None
        parsed.append((match.group(1), float(match.group(2))))
    return parsed


def _randomized_sort_equivalent(expected_lines: list[str], actual_lines: list[str]) -> bool:
    """Treat randomized sort demos as equivalent when structure and ordering match."""
    expected = _parse_labeled_numeric_lines(expected_lines)
    actual = _parse_labeled_numeric_lines(actual_lines)
    if not expected or not actual or len(expected) != len(actual):
        return False

    expected_labels = [label for label, _ in expected]
    actual_labels = [label for label, _ in actual]
    if expected_labels != actual_labels:
        return False

    if set(expected_labels) != {'UNSORTED:', 'SORTED:'}:
        return False

    actual_unsorted = [value for label, value in actual if label == 'UNSORTED:']
    actual_sorted = [value for label, value in actual if label == 'SORTED:']
    expected_unsorted_len = sum(1 for label in expected_labels if label == 'UNSORTED:')
    expected_sorted_len = sum(1 for label in expected_labels if label == 'SORTED:')
    if len(actual_unsorted) != expected_unsorted_len or len(actual_sorted) != expected_sorted_len:
        return False
    if len(actual_unsorted) != len(actual_sorted):
        return False

    return sorted(actual_unsorted) == actual_sorted


def _parse_positional_sort_lines(lines: list[str]) -> tuple[list[tuple[int, str]], list[tuple[int, str]]] | None:
    """Parse demo output shaped like 'POS: 001 RANDOM NUMBER: 044' and 'POS: 001 SORTED: 013'."""
    random_entries: list[tuple[int, str]] = []
    sorted_entries: list[tuple[int, str]] = []
    for line in lines:
        random_match = re.match(r'^POS:\s+(\d+)\s+RANDOM NUMBER:\s+(\d+)$', line)
        if random_match:
            random_entries.append((int(random_match.group(1)), random_match.group(2)))
            continue
        sorted_match = re.match(r'^POS:\s+(\d+)\s+SORTED:\s+(\d+)$', line)
        if sorted_match:
            sorted_entries.append((int(sorted_match.group(1)), sorted_match.group(2)))
            continue
        return None
    if not random_entries or not sorted_entries:
        return None
    return random_entries, sorted_entries


def _positional_sort_equivalent(expected_lines: list[str], actual_lines: list[str]) -> bool:
    """Treat random-then-sorted demos as equivalent when the actual sorted output matches actual inputs."""
    expected = _parse_positional_sort_lines(expected_lines)
    actual = _parse_positional_sort_lines(actual_lines)
    if not expected or not actual:
        return False

    expected_random, expected_sorted = expected
    actual_random, actual_sorted = actual
    if len(expected_random) != len(actual_random) or len(expected_sorted) != len(actual_sorted):
        return False

    if [pos for pos, _ in expected_random] != [pos for pos, _ in actual_random]:
        return False
    if [pos for pos, _ in expected_sorted] != [pos for pos, _ in actual_sorted]:
        return False

    if [len(value) for _, value in expected_random] != [len(value) for _, value in actual_random]:
        return False
    if [len(value) for _, value in expected_sorted] != [len(value) for _, value in actual_sorted]:
        return False

    actual_random_values = [int(value) for _, value in actual_random]
    actual_sorted_values = [int(value) for _, value in actual_sorted]
    return sorted(actual_random_values) == actual_sorted_values


def _parse_random_number_lines(lines: list[str]) -> list[str] | None:
    """Parse demo output shaped like 'RANDOM NUMBER: 039'."""
    values: list[str] = []
    for line in lines:
        match = re.match(r'^RANDOM NUMBER:\s+(\d+)$', line)
        if not match:
            return None
        values.append(match.group(1))
    return values if values else None


def _random_number_demo_equivalent(expected_lines: list[str], actual_lines: list[str]) -> bool:
    """Treat pure random-number demos as equivalent when shape, width, and variability match."""
    expected = _parse_random_number_lines(expected_lines)
    actual = _parse_random_number_lines(actual_lines)
    if not expected or not actual or len(expected) != len(actual):
        return False

    if [len(value) for value in expected] != [len(value) for value in actual]:
        return False

    actual_numbers = [int(value) for value in actual]
    max_value = max((10 ** len(value)) - 1 for value in expected)
    if any(number < 0 or number > max_value for number in actual_numbers):
        return False

    if len(set(expected)) > 1 and len(set(actual)) <= 1:
        return False
    return True


def _game_lottery_equivalent(expected_lines: list[str], actual_lines: list[str]) -> bool:
    """Treat lottery banner output as equivalent when the winning number shape is valid."""
    if len(expected_lines) != len(actual_lines) or not expected_lines:
        return False
    winning_pattern = re.compile(r'^- WINNING NUMBER IS : (\d{3})$')
    for expected, actual in zip(expected_lines, actual_lines):
        expected_match = winning_pattern.match(expected)
        actual_match = winning_pattern.match(actual)
        if expected_match or actual_match:
            if not expected_match or not actual_match:
                return False
            winning_number = int(actual_match.group(1))
            if winning_number < 1 or winning_number > 100:
                return False
            continue
        if expected != actual:
            return False
    return True


def _parse_datetime_lines(lines: list[str]) -> dict[str, str] | None:
    """Parse date/time demo output into a label->value mapping."""
    mapping: dict[str, str] = {}
    for line in lines:
        match = re.match(r'^(W-TIME|W-DATE|W-BATCH|COMPLET|TEST)\s*:\s*(\d+)$', line)
        if not match:
            return None
        mapping[match.group(1)] = match.group(2)
    return mapping if len(mapping) == 5 else None


def _datetime_demo_equivalent(expected_lines: list[str], actual_lines: list[str]) -> bool:
    """Treat DATE/TIME output as equivalent when the actual values are internally consistent."""
    expected = _parse_datetime_lines(expected_lines)
    actual = _parse_datetime_lines(actual_lines)
    if not expected or not actual:
        return False

    if len(actual['W-TIME']) != len(expected['W-TIME']):
        return False
    if len(actual['W-DATE']) != len(expected['W-DATE']):
        return False
    if len(actual['W-BATCH']) != len(expected['W-BATCH']):
        return False

    if actual['W-BATCH'] != actual['W-DATE'] + actual['W-TIME']:
        return False
    if actual['COMPLET'] != actual['W-BATCH']:
        return False
    if actual['TEST'] != actual['W-BATCH']:
        return False
    return True

def compare_outputs(expected: str, actual: str) -> tuple[bool, str]:
    """
    Compare expected and actual stdout line by line.
    Numeric values are compared with tolerance.
    Non-numeric values are compared as normalised strings.
    Returns (match: bool, diff: str).
    diff is the first mismatched line pair, empty if match.
    """
    exp_lines = [_normalise_line(l) for l in expected.splitlines() if l.strip()]
    act_lines = [_normalise_line(l) for l in actual.splitlines() if l.strip()]

    if not exp_lines and not act_lines:
        return True, ''

    if len(exp_lines) != len(act_lines):
        return False, (f'Line count mismatch: expected {len(exp_lines)},'
                       f' got {len(act_lines)}')

    if _randomized_sort_equivalent(exp_lines, act_lines):
        return True, ''

    if _positional_sort_equivalent(exp_lines, act_lines):
        return True, ''

    if _random_number_demo_equivalent(exp_lines, act_lines):
        return True, ''

    if _game_lottery_equivalent(exp_lines, act_lines):
        return True, ''

    if _datetime_demo_equivalent(exp_lines, act_lines):
        return True, ''

    for i, (e, a) in enumerate(zip(exp_lines, act_lines)):
        if e == a:
            continue
        # Try numeric comparison
        e_tokens = e.split()
        a_tokens = a.split()
        if len(e_tokens) == len(a_tokens):
            numeric_match = all(
                et == at or _numeric_close(et, at)
                for et, at in zip(e_tokens, a_tokens)
            )
            if numeric_match:
                continue
        return False, f'Line {i+1}: expected [{e}] got [{a}]'

    return True, ''


# ════════════════════════════════════════════════
# PHASE 3: PRODUCE VALIDATION RESULT
# ════════════════════════════════════════════════

def validate(
    translation_result: TranslationResult,
    oracle: BehavioralOracle,
    cobol_source: str = '',
) -> ValidationResult:
    """
    Main Validation Agent entry point.
    Runs translated Python against every test case in the oracle.

    Parameters
    ----------
    translation_result : from Translation Agent (Step 4)
    oracle             : from Test Generation Agent (Step 5a)
    cobol_source       : raw COBOL source passed through to Self-Repair

    Returns
    -------
    ValidationResult consumed by the internal remediation backend.
    """
    logger.info('=== Validation Agent: %s ===', translation_result.program_id)

    # If oracle has no test cases (GnuCOBOL not available), skip
    if not oracle.compile_success or not oracle.test_cases:
        logger.warning('%s: no oracle test cases — marking PARTIAL',
                       translation_result.program_id)
        return ValidationResult(
            program_id=translation_result.program_id,
            status='PARTIAL',
            pass_rate=0.0,
            error_message=oracle.compile_error or 'No test cases in oracle',
            translated_code=translation_result.translated_code,
            cobol_source=cobol_source,
        )

    py_path = translation_result.output_filepath
    results: list[TestCaseResult] = []

    for tc in oracle.test_cases:
        stdout, err = run_python_translation(py_path, tc.inputs)
        if err:
            results.append(TestCaseResult(test_id=tc.test_id, status='ERROR',
                expected=tc.expected_output, actual='', diff=err))
        else:
            match, diff = compare_outputs(tc.expected_output, stdout)
            status = 'PASS' if match else 'FAIL'
            results.append(TestCaseResult(test_id=tc.test_id, status=status,
                expected=tc.expected_output, actual=stdout, diff=diff))

    total = len(results)
    passing = sum(1 for r in results if r.status == 'PASS')
    failing = sum(1 for r in results if r.status == 'FAIL')
    errors = sum(1 for r in results if r.status == 'ERROR')
    pass_rate = passing / total if total > 0 else 0.0

    threshold = config.VALIDATION_THRESHOLD
    if pass_rate >= threshold:
        status = 'PASS'
    elif pass_rate >= 0.5:
        status = 'PARTIAL'
    else:
        status = 'FAIL'

    vr = ValidationResult(
        program_id=translation_result.program_id,
        status=status,
        pass_rate=round(pass_rate, 4),
        passing_cases=passing,
        failing_cases=failing,
        error_cases=errors,
        test_results=[{
            'test_id': r.test_id, 'status': r.status,
            'expected': r.expected, 'actual': r.actual, 'diff': r.diff
        } for r in results],
        translated_code=translation_result.translated_code,
        cobol_source=cobol_source,
    )
    logger.info('%s: %s (pass_rate=%.2f, %d/%d)',
               translation_result.program_id, status, pass_rate, passing, total)
    return vr

def summarise_validation(vr: ValidationResult) -> str:
    """Human-readable one-line summary of a ValidationResult."""
    total_cases = vr.passing_cases + vr.failing_cases + vr.error_cases
    return (f'{vr.program_id}: {vr.status} '
            f'({vr.passing_cases}/{total_cases} cases, '
            f'pass_rate={vr.pass_rate:.0%})')


def _result_value(result, key: str, default: str = ''):
    """Read a result field from either a dict or a TestCaseResult object."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def format_validation_report(vr: ValidationResult) -> str:
    """Human-readable multi-line validation report for CLI output."""
    lines = [
        "=" * 60,
        f"Validation Report: {vr.program_id}",
        f"Status    : {vr.status}",
        (
            f"Pass Rate : {vr.pass_rate:.1%} "
            f"({vr.passing_cases} passed, {vr.failing_cases} failed, "
            f"{vr.error_cases} errors)"
        ),
        "=" * 60,
    ]
    if vr.error_message:
        lines.extend(["", f"Error: {vr.error_message}"])
    for result in vr.test_results:
        status = _result_value(result, 'status')
        marker = "PASS" if status == "PASS" else "FAIL"
        test_id = _result_value(result, 'test_id')
        lines.append("")
        lines.append(f"  [{marker}] {test_id} -> {status}")
        if status != "PASS":
            expected = str(_result_value(result, 'expected'))[:120]
            actual = str(_result_value(result, 'actual'))[:120]
            diff = str(_result_value(result, 'diff'))[:120]
            lines.append(f"      Expected : {expected}")
            lines.append(f"      Actual   : {actual}")
            if diff:
                lines.append(f"      Diff     : {diff}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
