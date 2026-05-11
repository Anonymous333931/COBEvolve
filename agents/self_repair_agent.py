# ── FILE: cobol_moderniser/agents/self_repair_agent.py ────────────────
"""
agents/self_repair_agent.py — Internal remediation backend.

EXECUTE-phase remediation for the MAPE-K loop. This module is intentionally not
one of the six paper-facing COBEvolve agents; it is an implementation backend
invoked by the orchestrator when validation fails.
Grounded in: COBug bug localisation (ICSE 2026 under review)

Phases per retry attempt:
  1. KB similar-failure lookup
  2. Diagnose root cause via Ollama
  3. Generate corrected Python via Ollama
  4. Write fixed code + re-run Validation Agent
  5a. Success  -> save to KB, return FIXED
  5b. Retry    -> increment counter, adjust temperature
  5c. Rollback -> delete .py, log failure, return ROLLBACK
"""
from __future__ import annotations
import json
import logging, re, time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import config
from core.knowledge_base import (
    KnowledgeBase, TranslationRecord, FailureRecord, MigrationEvent
)
from utils.ollama_client import ollama_client, OllamaError
from agents.validation_agent import (
    validate, ValidationResult
)
from agents.translation_agent import TranslationResult
from agents.test_generation_agent import BehavioralOracle

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────
BASE_TEMPERATURE   = 0.02   # very low for first attempt (deterministic)
TEMP_INCREMENT     = 0.05   # increase per retry to add variation
SIMILAR_FAILURES_N = 3      # how many past failures to retrieve from ChromaDB
_ALLOWED_ERROR_TYPES = {
    'wrong_logic', 'missing_var', 'type_mismatch',
    'call_stub', 'numeric_format', 'other',
}

# ── Data models ──────────────────────────────────────────
@dataclass
class RepairDiagnosis:
    root_cause: str           # plain-English root cause
    error_type: str           # 'wrong_logic' | 'missing_var' | 'type_mismatch'
                              # | 'call_stub' | 'numeric_format' | 'other'
    fix_description: str      # what the fix should do
    affected_paragraphs: list = field(default_factory=list)
    confidence: float = 0.5

@dataclass
class RepairResult:
    program_id: str
    status: str               # 'FIXED' | 'FAILED' | 'ROLLBACK'
    attempts: int = 0
    final_pass_rate: float = 0.0
    diagnosis: str = ''
    fix_summary: str = ''
    output_filepath: str | None = None
    kb_failure_id: Optional[int] = None
    repaired_at: str = ''
    def __post_init__(self):
        if not self.repaired_at:
            self.repaired_at = datetime.now(timezone.utc).isoformat()


def _extract_json_block(text: str) -> str:
    """Return the first balanced JSON object substring, if any."""
    start = text.find('{')
    if start == -1:
        return ''
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if ch == '\\':
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ''


def _normalise_error_type(value: str) -> str:
    """Map model-provided labels to the supported repair taxonomy."""
    cleaned = (value or '').strip().lower()
    cleaned = re.sub(r'[^a-z_\-\s]', '', cleaned)
    cleaned = cleaned.replace('-', '_').replace(' ', '_')
    return cleaned if cleaned in _ALLOWED_ERROR_TYPES else 'other'


def _parse_diagnosis_response(raw: str) -> RepairDiagnosis | None:
    """Parse diagnosis output robustly even when the model returns JSON-like text."""
    candidates = [raw.strip()]
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE).strip()
    if cleaned and cleaned not in candidates:
        candidates.append(cleaned)
    block = _extract_json_block(cleaned)
    if block and block not in candidates:
        candidates.append(block)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and 'root_cause' in parsed:
            return RepairDiagnosis(
                root_cause=str(parsed.get('root_cause', 'unknown')),
                error_type=_normalise_error_type(str(parsed.get('error_type', 'other'))),
                fix_description=str(parsed.get('fix_description', '')),
                affected_paragraphs=list(parsed.get('affected_paragraphs', [])),
                confidence=float(parsed.get('confidence', 0.5)),
            )

    root_cause_m = re.search(r'"root_cause"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL)
    error_type_m = re.search(r'"error_type"\s*:\s*"?([a-zA-Z_]+)"?', cleaned)
    fix_desc_m = re.search(r'"fix_description"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL)
    affected_m = re.search(r'"affected_paragraphs"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
    confidence_m = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', cleaned)

    if not root_cause_m and not fix_desc_m:
        return None

    affected: list[str] = []
    if affected_m:
        affected = re.findall(r'"((?:[^"\\]|\\.)*)"', affected_m.group(1))

    def _unescape(value: str) -> str:
        return value.encode('utf-8').decode('unicode_escape')

    return RepairDiagnosis(
        root_cause=_unescape(root_cause_m.group(1)) if root_cause_m else 'unknown',
        error_type=_normalise_error_type(error_type_m.group(1) if error_type_m else 'other'),
        fix_description=_unescape(fix_desc_m.group(1)) if fix_desc_m else '',
        affected_paragraphs=[_unescape(v) for v in affected],
        confidence=float(confidence_m.group(1)) if confidence_m else 0.5,
    )

# PART B — Phase 1+2: KB Lookup + Diagnoser
# ════════════════════════════════════════════════
# PHASE 1: KB SIMILAR-FAILURE LOOKUP
# ════════════════════════════════════════════════

def _build_diff_summary(vr: ValidationResult) -> str:
    """Extract the most useful failing diffs as a compact text block."""
    lines = []
    for tr in vr.test_results[:5]:   # cap at 5 for prompt length
        if tr['status'] != 'PASS':
            lines.append(f"Test {tr['test_id']}: {tr['diff']}")
            if tr['expected']:
                lines.append(f"  expected: {tr['expected'][:120]}")
            if tr['actual']:
                lines.append(f"  actual:   {tr['actual'][:120]}")
    return '\n'.join(lines) if lines else 'No diff available'

def lookup_similar_failures(kb: KnowledgeBase, diff_summary: str) -> list[dict]:
    """
    Query ChromaDB for past failures similar to this diff.
    Returns list of {program_id, fix} dicts to use as few-shot examples.
    """
    try:
        results = kb.find_similar_failures(diff_summary, n=SIMILAR_FAILURES_N)
        logger.info('Found %d similar past failures in KB', len(results))
        return results
    except Exception as exc:
        logger.warning('KB similarity lookup failed: %s', exc)
        return []

# ════════════════════════════════════════════════
# PHASE 2: DIAGNOSE
# ════════════════════════════════════════════════

_DIAGNOSIS_SYSTEM = (
    'You are a senior COBOL-to-Python migration expert. '
    'You diagnose why a Python translation of COBOL code produces wrong output. '
    'You always respond with valid JSON only — no markdown, no preamble.'
)

_DIAGNOSIS_PROMPT = '''
A COBOL program has been translated to Python. The translation is failing
validation tests. Diagnose the root cause.

ORIGINAL COBOL:
{cobol_source}

TRANSLATED PYTHON (failing):
{translated_python}

FAILING TEST DIFFS:
{diff_summary}

{similar_context}

Return ONLY this JSON object:
{{
  "root_cause": "plain English description of why the translation fails",
  "error_type": "wrong_logic",
  "fix_description": "what specifically needs to change in the Python code",
  "affected_paragraphs": ["list", "of", "function", "names", "to", "fix"],
  "confidence": 0.0-1.0
}}

The "error_type" value must be exactly one of:
wrong_logic, missing_var, type_mismatch, call_stub, numeric_format, other
'''

def diagnose(
    vr: ValidationResult,
    similar_fixes: list[dict],
    temperature: float = BASE_TEMPERATURE,
) -> RepairDiagnosis:
    """
    Call Ollama to identify the root cause of the failing translation.
    Returns RepairDiagnosis. Falls back to generic diagnosis on error.
    """
    diff_summary = _build_diff_summary(vr)

    similar_context = ''
    if similar_fixes:
        ctx_lines = ['Similar past failures and their fixes:']
        for sf in similar_fixes[:3]:
            fix_text = sf.get('fix', 'unknown fix')[:200]
            ctx_lines.append(f'  - Fix applied: {fix_text}')
        similar_context = '\n'.join(ctx_lines)

    prompt = _DIAGNOSIS_PROMPT.format(
        cobol_source=vr.cobol_source[:3000],
        translated_python=vr.translated_code[:4000],
        diff_summary=diff_summary,
        similar_context=similar_context,
    )
    try:
        result = ollama_client.generate_json(
            model=config.MODEL_REPAIR,
            prompt=prompt,
            system=_DIAGNOSIS_SYSTEM,
            fallback={},
        )
        if isinstance(result, dict) and 'root_cause' in result:
            return RepairDiagnosis(
                root_cause=str(result.get('root_cause', 'unknown')),
                error_type=_normalise_error_type(str(result.get('error_type', 'other'))),
                fix_description=str(result.get('fix_description', '')),
                affected_paragraphs=list(result.get('affected_paragraphs', [])),
                confidence=float(result.get('confidence', 0.5)),
            )
    except OllamaError as exc:
        logger.warning('Ollama unavailable for diagnosis: %s', exc)
        return RepairDiagnosis(
            root_cause='Unknown — Ollama diagnosis unavailable',
            error_type='other',
            fix_description='Review failing diffs manually: ' + diff_summary[:200],
        )
    except Exception as exc:
        logger.warning('Diagnosis failed: %s', exc)
        return RepairDiagnosis(
            root_cause='Unknown — Ollama diagnosis unavailable',
            error_type='other',
            fix_description='Review failing diffs manually: ' + diff_summary[:200],
        )

    try:
        raw = ollama_client.generate(
            model=config.MODEL_REPAIR,
            prompt=prompt + "\n\nIMPORTANT: Return ONLY valid JSON.",
            system=_DIAGNOSIS_SYSTEM,
            temperature=temperature,
        )
        parsed = _parse_diagnosis_response(raw)
        if parsed is not None:
            return parsed
        logger.warning('Diagnosis parse failed; raw response begins: %s', raw[:200])
    except OllamaError as exc:
        logger.warning('Ollama unavailable for diagnosis: %s', exc)
    except Exception as exc:
        logger.warning('Diagnosis failed: %s', exc)

    # Fallback: generic diagnosis from diff
    return RepairDiagnosis(
        root_cause='Unknown — Ollama diagnosis unavailable',
        error_type='other',
        fix_description='Review failing diffs manually: ' + diff_summary[:200],
    )

# PART C — Phase 3: Fix Generator
# ════════════════════════════════════════════════
# PHASE 3: FIX GENERATOR
# ════════════════════════════════════════════════

_FIX_SYSTEM = (
    'You are a senior COBOL-to-Python migration engineer. '
    'You produce clean, correct Python 3.10+ code. '
    'You always respond with ONLY the complete corrected Python file. '
    'No explanations, no markdown fences, no preamble.'
)

_FIX_PROMPT = '''
Fix the Python translation of this COBOL program.

DIAGNOSIS:
Root cause: {root_cause}
Error type: {error_type}
What to fix: {fix_description}
Functions to fix: {affected_paragraphs}

ORIGINAL COBOL:
{cobol_source}

CURRENT PYTHON (broken):
{translated_python}

FAILING TEST DIFFS:
{diff_summary}

Requirements:
- Keep all existing imports and class structure
- Keep self.xxx variable names unchanged
- Fix ONLY the logic identified in the diagnosis
- Ensure DISPLAY statements produce output matching the COBOL stdout format
- For CALL stubs: implement the called program logic inline or via a helper method
- Return the complete corrected Python file, nothing else
'''

def generate_fix(
    vr: ValidationResult,
    diagnosis: RepairDiagnosis,
    temperature: float = BASE_TEMPERATURE,
) -> str:
    """
    Ask Ollama to produce a corrected Python translation.
    Returns the corrected source string.
    Falls back to the original translated_code if Ollama fails.
    """
    diff_summary = _build_diff_summary(vr)
    prompt = _FIX_PROMPT.format(
        root_cause=diagnosis.root_cause,
        error_type=diagnosis.error_type,
        fix_description=diagnosis.fix_description,
        affected_paragraphs=', '.join(diagnosis.affected_paragraphs) or 'all',
        cobol_source=vr.cobol_source[:3000],
        translated_python=vr.translated_code[:5000],
        diff_summary=diff_summary,
    )
    try:
        result = ollama_client.generate(
            model=config.MODEL_REPAIR,
            prompt=prompt,
            system=_FIX_SYSTEM,
            temperature=temperature,
        )
        # Strip markdown fences if model added them
        result = re.sub(r'^```python\n?', '', result, flags=re.MULTILINE)
        result = re.sub(r'^```\n?', '', result, flags=re.MULTILINE).strip()
        if len(result) > 50:   # sanity: must be non-trivial
            logger.info('Fix generated for %s (%d chars)',
                        vr.program_id, len(result))
            return result
        logger.warning('Fix generation returned trivial output for %s', vr.program_id)
    except OllamaError as exc:
        logger.warning('Ollama unavailable for fix generation: %s', exc)
    except Exception as exc:
        logger.warning('Fix generation failed: %s', exc)
    return vr.translated_code   # fallback: return original unchanged

# PART D — Phase 4+5: Retry Loop + KB Logger + Rollback
# ════════════════════════════════════════════════
# PHASE 4+5: RETRY LOOP, KB LOGGING, ROLLBACK
# ════════════════════════════════════════════════

def _save_success(
    kb: KnowledgeBase,
    vr: ValidationResult,
    program_id: str,
    final_python: str,
    output_filepath: str,
):
    """Save a successful translation to the Knowledge Base."""
    rec = TranslationRecord(
        cobol_hash=kb.hash_cobol(vr.cobol_source),
        program_id=program_id,
        cobol_code=vr.cobol_source,
        translated_code=final_python,
        output_filepath=output_filepath,
        language=config.TARGET_LANGUAGE,
        success=True,
        accuracy_score=vr.pass_rate,
    )
    kb.save_translation(rec)
    kb.log_event(MigrationEvent(
        program_id=program_id,
        module_path=output_filepath,
        status='validated',
        notes=f'Repaired: pass_rate={vr.pass_rate:.2f}',
    ))
    logger.info('%s: success saved to KB', program_id)

def _save_failure(
    kb: KnowledgeBase,
    vr: ValidationResult,
    diagnosis: RepairDiagnosis,
    fix_applied: str,
    resolved: bool,
) -> int:
    """Save a failure record to KB. Returns the row ID."""
    rec = FailureRecord(
        program_id=vr.program_id,
        cobol_code=vr.cobol_source,
        translated_code=vr.translated_code,
        error_message=_build_diff_summary(vr),
        diagnosis=diagnosis.root_cause,
        fix_applied=fix_applied[:500] if fix_applied else '',
        resolved=resolved,
    )
    row_id = kb.save_failure(rec)
    logger.info('%s: failure record saved (id=%d, resolved=%s)',
               vr.program_id, row_id, resolved)
    return row_id

def _rollback(output_filepath: str, program_id: str, kb: KnowledgeBase):
    """Delete the broken .py file and log ROLLED_BACK to migration_log."""
    p = Path(output_filepath)
    if p.exists() and p.is_file():
        p.unlink()
        logger.info('%s: rolled back — deleted %s', program_id, output_filepath)
    kb.log_event(MigrationEvent(
        program_id=program_id,
        module_path=output_filepath,
        status='rolled_back',
        notes='All repair retries exhausted',
    ))

# PART E — Main Entry Point: repair()
# ════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════

def repair(
    translation_result: TranslationResult,
    initial_vr: ValidationResult,
    oracle: BehavioralOracle,
    kb: KnowledgeBase | None = None,
    max_retries: int | None = None,
) -> RepairResult:
    """
    Main internal remediation entry point.

    Parameters
    ----------
    translation_result : from Translation Agent (Step 4)
    initial_vr         : from Validation Agent (Step 5) — FAIL or PARTIAL
    oracle             : BehavioralOracle for re-validation
    kb                 : KnowledgeBase instance (creates default if None)
    max_retries        : override config.MAX_REPAIR_RETRIES

    Returns
    -------
    RepairResult consumed by Orchestrator (Step 7)
    """
    config.setup()
    kb = kb or KnowledgeBase()
    max_r = max_retries if max_retries is not None else config.MAX_REPAIR_RETRIES
    pid = translation_result.program_id
    out_path = translation_result.output_filepath

    logger.info('=== Remediation backend: %s (max_retries=%d) ===', pid, max_r)

    # Precondition: if already PASS, nothing to do
    if initial_vr.status == 'PASS':
        logger.info('%s: already PASS — no repair needed', pid)
        return RepairResult(
            program_id=pid,
            status='FIXED',
            attempts=0,
            final_pass_rate=initial_vr.pass_rate,
            diagnosis='No repair needed',
            output_filepath=out_path,
        )

    current_vr = initial_vr
    last_diagnosis = RepairDiagnosis(
        root_cause='Not yet diagnosed', error_type='other', fix_description=''
    )
    last_fix_applied = ''
    kb_failure_id: Optional[int] = None

    for attempt in range(1, max_r + 1):
        logger.info('[%d/%d] Repair attempt for %s (pass_rate=%.2f)',
                    attempt, max_r, pid, current_vr.pass_rate)

        temperature = BASE_TEMPERATURE + (attempt - 1) * TEMP_INCREMENT

        # Phase 1: KB similar-failure lookup
        diff_summary = _build_diff_summary(current_vr)
        similar_fixes = lookup_similar_failures(kb, diff_summary)

        # Phase 2: Diagnose
        last_diagnosis = diagnose(
            current_vr, similar_fixes, temperature=temperature
        )
        logger.info('  Diagnosis: [%s] %s',
                    last_diagnosis.error_type, last_diagnosis.root_cause[:100])

        # Phase 3: Generate fix
        fixed_python = generate_fix(
            current_vr, last_diagnosis, temperature=temperature
        )
        last_fix_applied = last_diagnosis.fix_description

        # Phase 4: Write fixed code
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(fixed_python, encoding='utf-8')

        # Update TranslationResult with fixed code for re-validation
        fixed_tr = TranslationResult(
            program_id=translation_result.program_id,
            source_filepath=translation_result.source_filepath,
            output_filepath=out_path,
            translated_code=fixed_python,
            method='repair',
            confidence=0.0,
        )

        # Re-run Validation Agent
        new_vr = validate(
            fixed_tr, oracle, cobol_source=current_vr.cobol_source
        )
        logger.info('  After repair: %s (pass_rate=%.2f)',
                    new_vr.status, new_vr.pass_rate)

        # Phase 5a: Success
        if new_vr.pass_rate >= config.VALIDATION_THRESHOLD:
            _save_success(kb, new_vr, pid, fixed_python, out_path)
            # Also log the fix to failure KB so future runs can learn
            _save_failure(kb, initial_vr, last_diagnosis,
                          fix_applied=last_fix_applied, resolved=True)
            return RepairResult(
                program_id=pid,
                status='FIXED',
                attempts=attempt,
                final_pass_rate=new_vr.pass_rate,
                diagnosis=last_diagnosis.root_cause,
                fix_summary=last_fix_applied,
                output_filepath=out_path,
            )

        # Phase 5b: Not good enough yet — update current_vr and retry
        current_vr = ValidationResult(
            program_id=pid,
            status=new_vr.status,
            pass_rate=new_vr.pass_rate,
            passing_cases=new_vr.passing_cases,
            failing_cases=new_vr.failing_cases,
            error_cases=new_vr.error_cases,
            test_results=new_vr.test_results,
            translated_code=fixed_python,    # updated for next diagnosis
            cobol_source=initial_vr.cobol_source,
        )

    # Phase 5c: All retries exhausted — rollback
    logger.warning('%s: all %d retries exhausted — rolling back', pid, max_r)
    kb_failure_id = _save_failure(
        kb, current_vr, last_diagnosis,
        fix_applied=last_fix_applied, resolved=False
    )
    _rollback(out_path, pid, kb)
    return RepairResult(
        program_id=pid,
        status='ROLLBACK',
        attempts=max_r,
        final_pass_rate=current_vr.pass_rate,
        diagnosis=last_diagnosis.root_cause,
        fix_summary=last_fix_applied,
        output_filepath=None,
        kb_failure_id=kb_failure_id,
    )

def summarise_repair(rr: RepairResult) -> str:
    """Human-readable one-line summary of a RepairResult."""
    if rr.status == 'FIXED':
        return (f'{rr.program_id}: FIXED in {rr.attempts} attempt(s) '
                f'(pass_rate={rr.final_pass_rate:.0%})')
    elif rr.status == 'ROLLBACK':
        return (f'{rr.program_id}: ROLLBACK after {rr.attempts} attempt(s) '
                f'— {rr.diagnosis[:80]}')
    else:
        return f'{rr.program_id}: {rr.status} (pass_rate={rr.final_pass_rate:.0%})'
