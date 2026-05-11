# ── FILE: cobol_moderniser/agents/translation_agent.py ────────────
"""
agents/translation_agent.py — Translation Agent
EXECUTE phase of the MAPE‑K loop.
Grounded in: COB2PY (ICSME 2025), Java2COB (ICSE 2026)

Phases:
  1. KB Cache Check   — return cached result if available
  2. Rule-Based        — deterministic pattern translation
  3. Ollama LLM        — handle remaining complex constructs
  4. Save + Write      — persist to KB, write .py file
"""
from __future__ import annotations

import logging
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import config
from core.knowledge_base import KnowledgeBase, TranslationRecord
from utils.cobol_parser import COBOLProgram, DataItem, Paragraph, Statement
from utils.ollama_client import ollama_client, OllamaError
from agents.planning_agent import MigrationPlan, load_plan

logger = logging.getLogger(__name__)

_COBOL_VERB_RE = re.compile(
    r'^(STOP\s+RUN|GO\s+TO|NEXT\s+SENTENCE|MOVE|COMPUTE|PERFORM|IF|ELSE|'
    r'END-IF|EVALUATE|WHEN|END-EVALUATE|END-PERFORM|END-READ|END-STRING|CALL|DISPLAY|ACCEPT|ADD|SUBTRACT|MULTIPLY|DIVIDE|SET|STRING|OPEN|CLOSE|READ|EXIT|'
    r'STOP|GOBACK|INITIALIZE|CONTINUE)\b',
    re.IGNORECASE,
)

_COBOL_LITERAL_MAP = {
    'ZERO': '0',
    'ZEROS': '0',
    'ZEROES': '0',
    'SPACE': "''",
    'SPACES': "''",
}

_CURRENT_CONDITION_CHECKS: dict[str, str] = {}
_CURRENT_CONDITION_ASSIGNMENTS: dict[str, tuple[str, str]] = {}
_CURRENT_GROUP_FIELDS: set[str] = set()
_CURRENT_FIELD_TYPES: dict[str, str] = {}
_CURRENT_FIELD_PICS: dict[str, str] = {}
_TRANSLATION_CONTEXT_LOCK = threading.RLock()

# ── Translation confidence levels ─────────────────────────
CONF_RULE     = 1.00   # deterministic rule match
CONF_STUB     = 0.80   # stub generated (CALL, external ref)
CONF_LLM      = 0.75   # Ollama LLM handled it
CONF_CACHE    = 1.00   # retrieved from KB cache

# ── Output data model ──────────────────────────────
@dataclass
class TranslationResult:
    program_id:      str
    source_filepath: str
    output_filepath: str
    translated_code: str
    method:          str   = 'rule'   # rule|llm|hybrid|cache
    confidence:      float = 1.0
    rule_coverage:   float = 1.0      # fraction handled by rule engine
    warnings:        list  = field(default_factory=list)
    cobol_hash:      str   = ''


# ══════════════════════════════════════════════════════
# PHASE 2: RULE-BASED TRANSLATION ENGINE
# ══════════════════════════════════════════════════════

def _cobol_name_to_python(name: str) -> str:
    """COBOL-WS-NAME -> cobol_ws_name (snake_case)."""
    py_name = re.sub(r'[^a-z0-9_]+', '_', name.lower().replace('-', '_'))
    py_name = py_name.strip('_') or 'item'
    if py_name[0].isdigit():
        py_name = f'item_{py_name}'
    return py_name


def _normalise_cobol(text: str) -> str:
    """Collapse whitespace and uppercase for relaxed COBOL matching."""
    return ' '.join(text.split()).upper()


def _split_inline_statements(line: str) -> list[str]:
    """Split same-line COBOL statements only at true statement terminators."""
    parts: list[str] = []
    current: list[str] = []
    active_quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        current.append(ch)
        if active_quote is not None:
            if ch == active_quote:
                active_quote = None
            i += 1
            continue
        if ch in "'\"":
            active_quote = ch
            i += 1
            continue
        if ch == '.':
            j = i + 1
            while j < len(line) and line[j].isspace():
                j += 1
            remainder = line[j:]
            if remainder and _COBOL_VERB_RE.match(_normalise_cobol(remainder)):
                chunk = ''.join(current).strip()
                if chunk:
                    parts.append(chunk)
                current = []
                i = j
                continue
        i += 1
    tail = ''.join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _balanced_outer_parens(text: str) -> bool:
    depth = 0
    for i, ch in enumerate(text):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0 and i != len(text) - 1:
                return False
        if depth < 0:
            return False
    return depth == 0


def _strip_outer_parens(text: str) -> str:
    text = text.strip()
    while text.startswith('(') and text.endswith(')') and _balanced_outer_parens(text):
        text = text[1:-1].strip()
    return text


def _make_class_name(program_id: str) -> str:
    """Convert a COBOL program id into a valid Python class name."""
    parts = re.split(r'[^A-Za-z0-9]+', program_id)
    name = ''.join(part.capitalize() for part in parts if part)
    if not name:
        name = 'Program'
    if not name[0].isalpha():
        name = f'Prog{name}'
    return name


def _output_basename(program: COBOLProgram) -> str:
    """Prefer the source filename when it disambiguates duplicate PROGRAM-IDs."""
    program_slug = re.sub(r'[^a-z0-9]+', '-', program.program_id.lower()).strip('-')
    source_slug = re.sub(r'[^a-z0-9]+', '-', Path(program.filepath).stem.lower()).strip('-')
    return source_slug if source_slug and source_slug != program_slug else (program_slug or source_slug or 'module')


def _stash_strings(text: str) -> tuple[str, list[str]]:
    """Temporarily replace quoted strings so token rewrites do not mutate them."""
    literals: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        literals.append(match.group(0))
        return f'__STR{len(literals) - 1}__'

    return re.sub(r"'[^']*'|\"[^\"]*\"", _stash, text), literals


def _restore_strings(text: str, literals: list[str]) -> str:
    """Restore string literals hidden by _stash_strings."""
    for i, literal in enumerate(literals):
        text = text.replace(f'__STR{i}__', literal)
    return text


def _replace_qualified_names(expr: str) -> str:
    """Collapse COBOL qualified names like W-TIME OF W-BATCH to the leaf item."""
    previous = None
    result = expr
    while result != previous:
        previous = result
        result = re.sub(
            r'\b([A-Z][A-Z0-9\-_]*)\s+OF\s+([A-Z][A-Z0-9\-_]*)\b',
            lambda m: m.group(1),
            result,
            flags=re.IGNORECASE,
        )
    return result


def _python_literal(value: str) -> str:
    """Translate a COBOL VALUE/literal token into a Python literal."""
    raw = value.strip().rstrip('.')
    upper = raw.upper()
    if upper in _COBOL_LITERAL_MAP:
        return _COBOL_LITERAL_MAP[upper]
    if raw.startswith(("'", '"')):
        return repr(raw.strip("'\""))
    if re.fullmatch(r'-?\d+', raw):
        return raw
    if re.fullmatch(r'-?\d+[.,]\d+', raw):
        return f'Decimal("{raw.replace(",", ".")}")'
    return raw


def _replace_array_refs(expr: str, prefix_self: bool = True) -> str:
    """Translate COBOL array reads into safe Python helper calls."""
    pattern = re.compile(r'([A-Z][A-Z0-9\-_]*)\(([^()\n]+)\)')
    previous = None
    result = expr
    while result != previous:
        previous = result
        result = pattern.sub(
            lambda m: (
                f'{"self." if prefix_self else ""}_array_value('
                f'"{_cobol_name_to_python(m.group(1))}", '
                f'{_cobol_expr_to_python(m.group(2), prefix_self=prefix_self)})'
            ),
            result,
        )
    return result


def _replace_array_targets(expr: str, prefix_self: bool = True) -> str:
    """Translate COBOL array writes into Python list indexing."""
    pattern = re.compile(r'([A-Z][A-Z0-9\-_]*)\(([^()\n]+)\)')
    previous = None
    result = expr
    while result != previous:
        previous = result
        result = pattern.sub(
            lambda m: (
                f'{"self." if prefix_self else ""}{_cobol_name_to_python(m.group(1))}'
                f'[(max(1, int({_cobol_expr_to_python(m.group(2), prefix_self=prefix_self)})) - 1)]'
            ),
            result,
        )
    return result


def _replace_reference_mods(expr: str, prefix_self: bool = True) -> str:
    """Translate COBOL reference modification NAME(start:length) into Python slices."""
    pattern = re.compile(r'([A-Z][A-Z0-9\-_]*)\(([^():\n]+):([^()\n]+)\)')
    previous = None
    result = expr
    while result != previous:
        previous = result
        result = pattern.sub(
            lambda m: (
                f'{"self." if prefix_self else ""}{_cobol_name_to_python(m.group(1))}'
                f'[(int({_cobol_expr_to_python(m.group(2), prefix_self=prefix_self)}) - 1):'
                f'((int({_cobol_expr_to_python(m.group(2), prefix_self=prefix_self)}) - 1) + '
                f'int({_cobol_expr_to_python(m.group(3), prefix_self=prefix_self)}))]'
            ),
            result,
        )
    return result


def _replace_identifiers(expr: str, prefix_self: bool = True, expand_groups: bool = True) -> str:
    """Translate remaining COBOL identifiers into Python attribute refs."""
    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        upper = token.upper()
        if token.startswith('__STR') and token.endswith('__'):
            return token
        if upper in {'AND', 'OR', 'NOT', 'TRUE', 'FALSE'}:
            return token.lower()
        if upper in {'RANDOM', 'SECONDS', 'PAST', 'MIDNIGHT'}:
            return token
        if re.fullmatch(r'\d+', token):
            return token
        if expand_groups and upper in _CURRENT_GROUP_FIELDS:
            return f'{"self." if prefix_self else ""}_display_value("{_cobol_name_to_python(token)}")'
        return f'{"self." if prefix_self else ""}{_cobol_name_to_python(token)}'

    return re.sub(r'\b[A-Z][A-Z0-9\-_]*\b', _replace, expr)


def _cobol_expr_to_python(expr: str, prefix_self: bool = True) -> str:
    """Translate a COBOL arithmetic/value expression into Python."""
    expr = expr.strip().rstrip('.')
    if re.fullmatch(r'[+-]?\d+', expr):
        return str(int(expr))
    expr, literals = _stash_strings(expr)
    expr = _replace_qualified_names(expr)
    expr = re.sub(
        r"(?<![A-Z0-9_\"'])([+-]?\d+,\d+)\b",
        lambda m: f'Decimal("{m.group(1).replace(",", ".")}")',
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(
        r"(?<![A-Z0-9_\"'])([+-]?\d+\.\d+)\b",
        lambda m: f'Decimal("{m.group(1)}")',
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(
        r'/\s*(\d+)\b',
        lambda m: f'/ Decimal("{m.group(1)}")',
        expr,
    )
    for cobol_literal, python_literal in _COBOL_LITERAL_MAP.items():
        expr = re.sub(
            rf'\b{cobol_literal}\b',
            python_literal,
            expr,
            flags=re.IGNORECASE,
        )
    expr = re.sub(
        r'FUNCTION\s+RANDOM\s*\(\s*FUNCTION\s+SECONDS-PAST-MIDNIGHT\s*\)',
        '_cobol_random((int(time.time()) % 86400))',
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(
        r'FUNCTION\s+SECONDS-PAST-MIDNIGHT',
        '(int(time.time()) % 86400)',
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(
        r'FUNCTION\s+RANDOM\s*\(\s*([^)]+?)\s*\)',
        lambda m: f'_cobol_random({_cobol_expr_to_python(m.group(1), prefix_self=prefix_self)})',
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(r'FUNCTION\s+RANDOM\b', 'random.random()', expr, flags=re.IGNORECASE)
    expr = _replace_reference_mods(expr, prefix_self=prefix_self)
    expr = _replace_array_refs(expr, prefix_self=prefix_self)
    expr = _replace_identifiers(expr, prefix_self=prefix_self, expand_groups=True)
    expr = ' '.join(expr.split())
    return _restore_strings(expr, literals)


def _cobol_target_to_python(target: str) -> str:
    """Translate a COBOL assignment target, including array references."""
    target = target.strip().rstrip('.')
    target, literals = _stash_strings(target)
    target = _replace_qualified_names(target)
    target = _replace_reference_mods(target, prefix_self=True)
    target = _replace_array_targets(target, prefix_self=True)
    target = _replace_identifiers(target, prefix_self=True, expand_groups=False)
    target = ' '.join(target.split())
    return _restore_strings(target, literals)


def _cobol_cond_to_python(condition: str) -> str:
    """Convert common COBOL condition syntax to Python."""
    cond = _strip_outer_parens(condition)
    cond, string_literals = _stash_strings(cond)

    def _replace_condition_name(match: re.Match[str]) -> str:
        token = match.group(0).upper()
        expr = _CURRENT_CONDITION_CHECKS.get(token)
        return expr if expr is not None else match.group(0)

    def _replace_is_test(pattern: str, build):
        nonlocal cond
        cond = re.sub(pattern, build, cond, flags=re.IGNORECASE)

    _replace_is_test(
        r'([A-Z][A-Z0-9\-_]*(?:\([^)\n]+\))?)\s+IS\s+NOT\s+ALPHABETIC\b',
        lambda m: f'(not self._is_alphabetic({_cobol_expr_to_python(m.group(1))}))',
    )
    _replace_is_test(
        r'([A-Z][A-Z0-9\-_]*(?:\([^)\n]+\))?)\s+IS\s+ALPHABETIC\b',
        lambda m: f'self._is_alphabetic({_cobol_expr_to_python(m.group(1))})',
    )
    _replace_is_test(
        r'([A-Z][A-Z0-9\-_]*(?:\([^)\n]+\))?)\s+IS\s+NOT\s+NUMERIC\b',
        lambda m: f'(not self._is_numeric({_cobol_expr_to_python(m.group(1))}))',
    )
    _replace_is_test(
        r'([A-Z][A-Z0-9\-_]*(?:\([^)\n]+\))?)\s+IS\s+NUMERIC\b',
        lambda m: f'self._is_numeric({_cobol_expr_to_python(m.group(1))})',
    )
    _replace_is_test(
        r'([A-Z][A-Z0-9\-_]*(?:\([^)\n]+\))?)\s+IS\s+POSITIVE\b',
        lambda m: f'({_cobol_expr_to_python(m.group(1))} > 0)',
    )
    _replace_is_test(
        r'([A-Z][A-Z0-9\-_]*(?:\([^)\n]+\))?)\s+IS\s+NEGATIVE\b',
        lambda m: f'({_cobol_expr_to_python(m.group(1))} < 0)',
    )
    _replace_is_test(
        r'([A-Z][A-Z0-9\-_]*(?:\([^)\n]+\))?)\s+IS\s+ZERO\b',
        lambda m: f'({_cobol_expr_to_python(m.group(1))} == 0)',
    )
    cond = re.sub(r'\b[A-Z][A-Z0-9\-_]*\b', _replace_condition_name, cond)
    replacements = [
        (r'\bNOT\s*=', '!='),
        (r'\bNOT\s+EQUAL(?:\s+TO)?\b', '!='),
        (r'\bGREATER\s+THAN\s+OR\s+EQUAL\s+TO\b', '>='),
        (r'\bLESS\s+THAN\s+OR\s+EQUAL\s+TO\b', '<='),
        (r'\bGREATER\s+THAN\b', '>'),
        (r'\bLESS\s+THAN\b', '<'),
        (r'\bEQUAL(?:\s+TO)?\b', '=='),
        (r'(?<![<>=!])=(?!=)', '=='),
        (r'\bIS\b', ''),
        (r'\bTHEN\b', ''),
        (r'\bAND\b', 'and'),
        (r'\bOR\b', 'or'),
        (r'\bNOT\b', 'not'),
    ]
    for pattern, repl in replacements:
        cond = re.sub(pattern, repl, cond, flags=re.IGNORECASE)
    cond, injected_literals = _stash_strings(cond)
    cond = _replace_array_refs(cond)
    cond = _replace_identifiers(cond)
    cond = ' '.join(cond.split())
    cond = _restore_strings(cond, injected_literals)
    return _restore_strings(cond, string_literals)


def _dedup_imports(source: str) -> str:
    """Remove duplicate import lines while preserving other source lines."""
    seen_imports: set[str] = set()
    lines: list[str] = []
    for line in source.splitlines():
        if line.startswith(('import ', 'from ')):
            if line in seen_imports:
                continue
            seen_imports.add(line)
        lines.append(line)
    return '\n'.join(lines)


def _pic_to_python_default(pic: Optional[str]) -> str:
    """Return a Python default value matching the PIC clause."""
    if not pic:
        return 'None'
    p = pic.upper().replace(' ', '').rstrip('.')
    if p == 'POINTER':
        return 'None'
    if p.startswith(('X', 'A')):
        return "''"
    if any(token in p for token in ('Z', '/', 'CR', 'DB', '$', '*')):
        return "''"
    if 'V' in p or '.' in p:
        return 'Decimal("0")'
    return '0'


def _pic_to_python_type(pic: Optional[str]) -> str:
    """Return a Python type annotation for a PIC clause."""
    if not pic:
        return 'Any'
    p = pic.upper().replace(' ', '').rstrip('.')
    if p == 'POINTER':
        return 'Any'
    if p.startswith(('X', 'A')):
        return 'str'
    if any(token in p for token in ('Z', '/', 'CR', 'DB', '$', '*')):
        return 'str'
    if 'V' in p or '.' in p:
        return 'Decimal'
    return 'int'


def _needs_display_support(program: COBOLProgram) -> bool:
    """Return True when generated DISPLAY output depends on PIC formatting."""
    return bool(re.search(r'\bPIC(?:TURE)?\s+(?:IS\s+)?[SX9Z]', program.raw_source, re.IGNORECASE))


def _infer_value_type(value: Optional[str]) -> str:
    """Infer a Python type when PIC is absent but VALUE is present."""
    if not value:
        return 'Any'
    raw = value.strip().strip('.').strip("'\"")
    if re.fullmatch(r'-?\d+', raw):
        return 'int'
    if re.fullmatch(r'-?\d+\.\d+', raw):
        return 'Decimal'
    return 'str'


def _default_list_factory(py_default: str, size_expr: str | None) -> str:
    """Build a dataclass default_factory for OCCURS items."""
    if size_expr:
        return f'field(default_factory=lambda: [{py_default}] * {size_expr})'
    return 'field(default_factory=list)'


def _resolve_occurs_size(item: DataItem, constants: dict[str, str]) -> str | None:
    """Resolve OCCURS size, including symbolic level-78 constants."""
    if item.occurs is None:
        return None
    if isinstance(item.occurs, int):
        return str(item.occurs)
    return constants.get(str(item.occurs).upper())


def _walk_data_items(items: list[DataItem]) -> list[DataItem]:
    """Flatten nested data items while preserving source order."""
    flattened: list[DataItem] = []
    for item in items:
        flattened.append(item)
        if item.children:
            flattened.extend(_walk_data_items(item.children))
    return flattened


def _build_unique_item_names(items: list[DataItem]) -> dict[int, str]:
    """Assign unique Python names to data items, including repeated FILLER fields."""
    names: dict[int, str] = {}
    seen: dict[str, int] = {}

    def _visit(nodes: list[DataItem]) -> None:
        for item in nodes:
            base = _cobol_name_to_python(item.name)
            if item.name.upper() == 'FILLER':
                base = 'filler'
            count = seen.get(base, 0) + 1
            seen[base] = count
            names[id(item)] = base if count == 1 else f'{base}_{count}'
            if item.children:
                _visit(item.children)

    _visit(items)
    return names


def _collect_leaf_names(item: DataItem, item_names: dict[int, str]) -> list[str]:
    """Collect leaf field python names for a group item."""
    if not item.children:
        return [item_names[id(item)]]
    names: list[str] = []
    for child in item.children:
        names.extend(_collect_leaf_names(child, item_names))
    return names


def _collect_leaf_occurs_sizes(
    items: list[DataItem],
    constants: dict[str, str],
    item_names: dict[int, str],
) -> dict[str, str]:
    """Record OCCURS sizes for leaves, including leaves inside OCCURS groups."""
    sizes: dict[str, str] = {}

    def _visit(nodes: list[DataItem], inherited_size: str | None = None) -> None:
        for item in nodes:
            current_size = _resolve_occurs_size(item, constants) or inherited_size
            if not item.children:
                if current_size:
                    sizes[item_names[id(item)]] = current_size
                continue
            _visit([child for child in item.children if child.level != 88], current_size)

    _visit(items)
    return sizes


def _collect_group_field_specs(
    item: DataItem,
    item_names: dict[int, str],
    constants: dict[str, str],
    inherited_index: int | None = None,
) -> list[dict[str, int | str]]:
    """Collect fixed-width leaf specs for assigning into a group item."""
    children = [child for child in item.children if child.level != 88]
    if not children:
        length = _item_storage_length(item)
        if not length:
            return []
        occurs_size = _resolve_occurs_size(item, constants)
        if occurs_size and occurs_size.isdigit():
            return [
                {
                    'name': item_names[id(item)],
                    'length': length,
                    'index': idx,
                }
                for idx in range(int(occurs_size))
            ]
        spec: dict[str, int | str] = {
            'name': item_names[id(item)],
            'length': length,
        }
        if inherited_index is not None:
            spec['index'] = inherited_index
        return [spec]

    specs: list[dict[str, int | str]] = []
    occurs_size = _resolve_occurs_size(item, constants)
    if occurs_size and occurs_size.isdigit():
        for idx in range(int(occurs_size)):
            for child in children:
                specs.extend(
                    _collect_group_field_specs(
                        child,
                        item_names,
                        constants,
                        inherited_index=idx,
                    )
                )
        return specs
    for child in children:
        specs.extend(_collect_group_field_specs(child, item_names, constants, inherited_index))
    return specs


def _field_base_type(item: DataItem) -> str:
    """Return the logical scalar type name for a data item."""
    if item.pic:
        return _pic_to_python_type(item.pic)
    if item.value:
        return _infer_value_type(item.value)
    return 'Any'


def _target_field_name(cobol_target: str) -> str:
    """Extract the base field name from a COBOL assignment target."""
    target = _replace_qualified_names(cobol_target.strip().rstrip('.'))
    return _cobol_name_to_python(target.split('(', 1)[0].strip())


def _simple_source_display_expr(cobol_source: str) -> str | None:
    """Render a simple source token using its COBOL display picture."""
    source = _replace_qualified_names(cobol_source.strip().rstrip('.'))
    if source.upper() in _COBOL_LITERAL_MAP:
        return None
    identifier = re.fullmatch(r'([A-Z][A-Z0-9\-_]*)', source, re.IGNORECASE)
    if identifier:
        field_name = _cobol_name_to_python(identifier.group(1))
        return f'self._format_display("{field_name}", self.{field_name})'

    array_ref = re.fullmatch(r'([A-Z][A-Z0-9\-_]*)\(([^)\n]+)\)', source, re.IGNORECASE)
    if array_ref:
        field_name = _cobol_name_to_python(array_ref.group(1))
        return f'self._format_display("{field_name}", {_cobol_expr_to_python(source)})'

    return None


def _literal_for_parent_type(raw: str, parent_type: str) -> str:
    """Convert an 88-level literal into Python syntax compatible with the parent type."""
    value = raw.strip().strip('.')
    upper_value = value.strip("'\"").upper()
    if upper_value in ('ZERO', 'ZEROS', 'ZEROES'):
        normalized = '0'
    elif upper_value in ('SPACE', 'SPACES'):
        normalized = ' '
    else:
        normalized = value.strip("'\"")
    if parent_type == 'Decimal':
        return f'Decimal("{normalized}")'
    if parent_type == 'int':
        return str(int(normalized))
    return repr(normalized)


def _primary_condition_value(raw: str) -> str:
    """Return the primary 88-level VALUE clause, ignoring FALSE/TRUE aliases."""
    primary = re.split(r'\bFALSE\s+IS\b|\bTRUE\s+IS\b', raw, maxsplit=1, flags=re.IGNORECASE)[0]
    return primary.strip()


def _extract_condition_checks(program: COBOLProgram) -> dict[str, str]:
    """Extract 88-level condition-name checks as Python expressions."""
    checks: dict[str, str] = {}
    for item in _walk_data_items(program.all_data_items):
        if not item.children:
            continue
        parent_name = _cobol_name_to_python(item.name)
        parent_type = _field_base_type(item)
        for child in item.children:
            if child.level != 88 or not child.value:
                continue
            cond_name = child.name.upper()
            spec = _primary_condition_value(child.value).upper()
            if ' THRU ' in spec or ' THROUGH ' in spec:
                parts = re.split(r'\s+THRU(?:OUGH)?\s+', _primary_condition_value(child.value), maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    low = _literal_for_parent_type(parts[0], parent_type)
                    high = _literal_for_parent_type(parts[1], parent_type)
                    checks[cond_name] = f'({low} <= self.{parent_name} <= {high})'
                    continue
            values = [
                part.strip()
                for part in re.split(r'\s*,\s*|\s+OR\s+', _primary_condition_value(child.value), flags=re.IGNORECASE)
                if part.strip()
            ]
            if len(values) > 1:
                rendered = ', '.join(_literal_for_parent_type(value, parent_type) for value in values)
                checks[cond_name] = f'(self.{parent_name} in ({rendered},))'
            else:
                literal = _literal_for_parent_type(values[0], parent_type)
                checks[cond_name] = f'(self.{parent_name} == {literal})'
    return checks


def _extract_condition_assignments(program: COBOLProgram) -> dict[str, tuple[str, str]]:
    """Map 88-level condition names to a concrete parent assignment."""
    assignments: dict[str, tuple[str, str]] = {}
    for item in _walk_data_items(program.all_data_items):
        if not item.children:
            continue
        parent_name = _cobol_name_to_python(item.name)
        parent_type = _field_base_type(item)
        for child in item.children:
            if child.level != 88 or not child.value:
                continue
            parts = re.split(
                r'\s+THRU(?:OUGH)?\s+|\s*,\s*|\s+OR\s+',
                _primary_condition_value(child.value),
                maxsplit=1,
                flags=re.IGNORECASE,
            )
            if not parts:
                continue
            assignments[child.name.upper()] = (
                parent_name,
                _literal_for_parent_type(parts[0], parent_type),
            )
    return assignments


def _pic_storage_length(pic: str | None) -> int:
    """Approximate the on-disk length of a COBOL PIC clause."""
    if not pic:
        return 0
    expanded = pic.upper().replace(' ', '')
    while re.search(r'([X9ZAS])\((\d+)\)', expanded):
        expanded = re.sub(
            r'([X9ZAS])\((\d+)\)',
            lambda m: m.group(1) * int(m.group(2)),
            expanded,
        )
    length = 0
    for ch in expanded:
        if ch in {'X', '9', 'Z', 'A'}:
            length += 1
    return length


def _item_storage_length(item: DataItem) -> int:
    """Compute a fixed-width length for a file-section item."""
    if item.children:
        return sum(
            _item_storage_length(child)
            for child in item.children
            if child.level != 88
        )
    return _pic_storage_length(item.pic)


def _collect_file_leaf_specs(item: DataItem, offset: int = 0) -> tuple[list[dict[str, int | str]], int]:
    """Collect leaf field slices for a file record group."""
    fields: list[dict[str, int | str]] = []
    children = [child for child in item.children if child.level != 88]
    if not children:
        length = _item_storage_length(item)
        if length:
            fields.append({
                'name': _cobol_name_to_python(item.name),
                'start': offset,
                'length': length,
            })
        return fields, offset + length
    current = offset
    for child in children:
        child_fields, current = _collect_file_leaf_specs(child, current)
        fields.extend(child_fields)
    return fields, current


def _extract_file_specs(program: COBOLProgram) -> dict[str, dict[str, object]]:
    """Extract sequential file metadata from FILE-CONTROL and FILE SECTION."""
    specs: dict[str, dict[str, object]] = {}
    lines = [line.rstrip() for line in program.raw_source.splitlines()]
    i = 0
    while i < len(lines):
        line = ' '.join(lines[i].split())
        if not re.match(r'^SELECT\s+[A-Z][A-Z0-9\-]*\b', line, re.IGNORECASE):
            i += 1
            continue
        block = [line]
        i += 1
        while i < len(lines):
            current = ' '.join(lines[i].split())
            if re.match(
                r'^(SELECT\b|[A-Z][A-Z0-9\-]*\s+SECTION\.|DATA\s+DIVISION\.|PROCEDURE\s+DIVISION\.)',
                current,
                re.IGNORECASE,
            ):
                break
            if current:
                block.append(current)
            i += 1
        block_text = ' '.join(block)
        select_m = re.match(r'SELECT\s+([A-Z][A-Z0-9\-]*)', block_text, re.IGNORECASE)
        assign_m = re.search(r'ASSIGN\s+TO\s+("[^"]*"|\'[^\']*\'|\S+)', block_text, re.IGNORECASE)
        if not select_m or not assign_m:
            continue
        file_name = select_m.group(1)
        specs[_cobol_name_to_python(file_name)] = {
            'assign_to': assign_m.group(1).strip().strip('"\''),
            'status_var': '',
            'record_group': '',
            'fields': [],
            'base_dir': str(Path(program.filepath).resolve().parent),
        }
        status_m = re.search(r'FILE\s+STATUS\s+([A-Z][A-Z0-9\-]*)', block_text, re.IGNORECASE)
        if status_m:
            specs[_cobol_name_to_python(file_name)]['status_var'] = _cobol_name_to_python(status_m.group(1))
        continue

    file_section_text = ''
    file_section_match = re.search(
        r'FILE\s+SECTION\.(.+?)(?=WORKING-STORAGE\s+SECTION\.|LINKAGE\s+SECTION\.|PROCEDURE\s+DIVISION\.|\Z)',
        program.raw_source,
        re.IGNORECASE | re.DOTALL,
    )
    if file_section_match:
        file_section_text = file_section_match.group(1)
    current_file = ''
    for raw_line in file_section_text.splitlines():
        line = ' '.join(raw_line.split())
        if not line:
            continue
        fd_m = re.match(r'FD\s+([A-Z][A-Z0-9\-]*)\.?', line, re.IGNORECASE)
        if fd_m:
            current_file = _cobol_name_to_python(fd_m.group(1))
            specs.setdefault(current_file, {
                'assign_to': '',
                'status_var': '',
                'record_group': '',
                'fields': [],
                'base_dir': str(Path(program.filepath).resolve().parent),
            })
            continue
        group_m = re.match(r'01\s+([A-Z][A-Z0-9\-]*)', line, re.IGNORECASE)
        if current_file and group_m and not specs[current_file].get('record_group'):
            specs[current_file]['record_group'] = _cobol_name_to_python(group_m.group(1))

    file_items = {
        _cobol_name_to_python(item.name): item
        for item in program.file_section
    }
    for spec in specs.values():
        record_group = spec.get('record_group')
        if not record_group:
            continue
        record_item = file_items.get(str(record_group))
        if not record_item:
            continue
        fields, _ = _collect_file_leaf_specs(record_item, 0)
        spec['fields'] = fields
    return specs


def translate_data_division(program: COBOLProgram) -> str:
    """
    Translate DATA DIVISION (FILE + WORKING-STORAGE + LINKAGE SECTION)
    into a Python dataclass.
    """
    flat_items = _walk_data_items(program.all_data_items)
    item_names = _build_unique_item_names(program.all_data_items)
    file_specs = _extract_file_specs(program)
    paragraph_order = [_cobol_name_to_python(paragraph.name) for paragraph in program.paragraphs]
    group_children = {
        item_names[id(item)]: _collect_leaf_names(item, item_names)
        for item in flat_items
        if item.children and any(child.level != 88 for child in item.children)
    }
    constants: dict[str, str] = {}
    for item in flat_items:
        if item.level == 78 and item.value:
            constants[item.name.upper()] = _python_literal(item.value)
    group_field_specs = {
        item_names[id(item)]: _collect_group_field_specs(item, item_names, constants)
        for item in flat_items
        if item.children and any(child.level != 88 for child in item.children)
    }
    field_types = {
        item_names[id(item)]: _field_base_type(item)
        for item in flat_items
    }
    field_pics = {
        item_names[id(item)]: item.pic
        for item in flat_items
        if item.pic
    }
    pointer_fields = {
        item_names[id(item)]
        for item in flat_items
        if (item.pic or '').upper().replace(' ', '').rstrip('.') == 'POINTER'
    }
    leaf_occurs_sizes = _collect_leaf_occurs_sizes(program.all_data_items, constants, item_names)
    raw_name_to_unique = {
        item.name.upper().rstrip('.'): item_names[id(item)]
        for item in flat_items
    }
    redefines_map = {
        item_names[id(item)]: raw_name_to_unique.get(item.redefines.upper().rstrip('.'))
        for item in flat_items
        if item.redefines and raw_name_to_unique.get(item.redefines.upper().rstrip('.'))
    }
    decimal_separator = ',' if re.search(
        r'DECIMAL-POINT\s+IS\s+COMMA',
        program.raw_source,
        re.IGNORECASE,
    ) else '.'

    lines = [
        'from __future__ import annotations',
        'import os',
        'import re',
        'import random',
        'import sys',
        'import time',
        'from decimal import Decimal, ROUND_DOWN',
        'from dataclasses import dataclass, field',
        'from pathlib import Path',
        'from typing import Any',
        '',
        '# cobol_moderniser: runtime-support-v13',
        '',
        'def _cobol_random(seed=None):',
        '    if seed is not None:',
        '        random.seed(int(seed))',
        '    return random.random()',
        '',
        '',
        f'@dataclass',
        f'class {_make_class_name(program.program_id)}:',
    ]
    if not flat_items:
        lines.append('    pass')
        return '\n'.join(lines)

    for item in flat_items:
        if item.level in (66, 88):
            continue
        py_name = item_names[id(item)]
        py_type = _pic_to_python_type(item.pic)
        py_def = _pic_to_python_default(item.pic)
        if item.level == 78 and not item.pic:
            py_type = _infer_value_type(item.value)
            py_def = 'None'
        occurs_size = leaf_occurs_sizes.get(py_name)
        if occurs_size and not item.children:
            elem_type = py_type if py_type != 'Any' else _infer_value_type(item.value)
            elem_default = py_def if py_def != 'None' else _python_literal(item.value or '0')
            lines.append(
                f'    {py_name}: list[{elem_type}] = '
                f'{_default_list_factory(elem_default, occurs_size)}'
            )
            continue
        if item.value and item.value not in ('ZERO', 'ZEROS', 'ZEROES',
                                             'SPACES', 'SPACE', 'LOW-VALUE',
                                             'HIGH-VALUE'):
            py_def = _python_literal(item.value)
        lines.append(f'    {py_name}: {py_type} = {py_def}')

    lines.extend([
        '    _file_handles: dict[str, Any] = field(default_factory=dict, init=False, repr=False)',
        '',
    ])

    lines.extend([
        '',
        f'    _field_types = {repr(field_types)}',
        f'    _field_pics = {repr(field_pics)}',
        f'    _group_children = {repr(group_children)}',
        f'    _group_field_specs = {repr(group_field_specs)}',
        f'    _redefines = {repr(redefines_map)}',
        f'    _pointer_fields = {repr(sorted(pointer_fields))}',
        f'    _decimal_separator = {repr(decimal_separator)}',
        f'    _file_specs = {repr(file_specs)}',
        f'    _paragraph_order = {repr(paragraph_order)}',
        '',
        '    def __post_init__(self):',
        '        for group_name in self._group_children:',
        '            if getattr(self, group_name, None) is None:',
        '                setattr(self, group_name, self._display_value(group_name))',
        '        for alias_name, base_name in self._redefines.items():',
        '            self._assign_group_value(alias_name, self._display_value(base_name))',
        '',
        '    def _coerce_value(self, name: str, value):',
        "        field_type = self._field_types.get(name, 'Any')",
        "        if field_type == 'int':",
        '            if value is None:',
        '                return 0',
        "            if isinstance(value, bool):",
        '                return int(value)',
        '            if isinstance(value, Decimal):',
        '                return int(value)',
        '            text = str(value).strip()',
        '            if not text:',
        '                return 0',
        "            if text.count(',') == 1 and '.' not in text:",
        "                text = text.replace(',', '.')",
        '            pic = self._field_pics.get(name)',
        '            try:',
        '                number = int(text)',
        '            except ValueError:',
        '                number = int(Decimal(text))',
        '            if pic:',
        '                expanded = self._expand_pic(pic)',
        "                numeric_picture = expanded.lstrip('S')",
        "                if numeric_picture and set(numeric_picture) <= {'9'}:",
        '                    width = len(numeric_picture)',
        '                    number = abs(number) % (10 ** max(width, 1))',
        '            return number',
        "        if field_type == 'Decimal':",
        '            if isinstance(value, Decimal):',
        '                decimal_value = value',
        '            else:',
        '                text = str(value).strip()',
        '                if not text:',
        '                    return Decimal("0")',
        "                if text.count(',') == 1 and '.' not in text:",
        "                    text = text.replace(',', '.')",
        '                decimal_value = Decimal(text)',
        '            pic = self._field_pics.get(name)',
        '            if pic:',
        '                decimal_value = self._coerce_decimal_to_pic(decimal_value, pic)',
        '            return decimal_value',
        "        if field_type == 'str':",
        "            return '' if value is None else str(value)",
        '        return value',
        '',
        '    def _read_stdin(self) -> str:',
        '        try:',
        '            return input()',
        '        except EOFError:',
        "            return ''",
        '',
        '    def _coerce_decimal_to_pic(self, decimal_value: Decimal, pic: str) -> Decimal:',
        '        expanded = self._expand_pic(pic)',
        "        numeric_picture = expanded.lstrip('S')",
        "        if 'V' not in numeric_picture and '.' not in numeric_picture:",
        '            return decimal_value',
        "        separator = 'V' if 'V' in numeric_picture else '.'",
        '        integer_digits, fractional_digits = numeric_picture.split(separator, 1)',
        '        scale = len(fractional_digits)',
        '        width = len(integer_digits) + scale',
        '        scaled = int((abs(decimal_value) * (10 ** scale)).to_integral_value(rounding=ROUND_DOWN))',
        '        if width > 0:',
        '            scaled %= 10 ** width',
        '        whole = scaled // (10 ** scale) if scale else scaled',
        '        frac = scaled % (10 ** scale) if scale else 0',
        "        text = str(whole) if scale == 0 else f'{whole}.{str(frac).zfill(scale)}'",
        '        result = Decimal(text)',
        '        if scale:',
        "            quant = Decimal('1.' + ('0' * scale))",
        '            result = result.quantize(quant)',
        "        if expanded.startswith('S') and decimal_value < 0:",
        '            result = -result',
        '        return result',
        '',
        '    def _accept_input(self, name: str) -> str:',
        '        raw = self._read_stdin()',
        '        pic = self._field_pics.get(name)',
        '        if not pic:',
        '            return raw',
        '        expanded = self._expand_pic(pic)',
        "        numeric_picture = expanded.lstrip('S')",
        "        if numeric_picture and set(numeric_picture) <= {'9'}:",
        "            digits = ''.join(ch for ch in raw if ch.isdigit())",
        '            if not digits:',
        '                return raw',
        '            return digits[:len(numeric_picture)]',
        "        if 'V' in numeric_picture or '.' in numeric_picture:",
        "            separator = 'V' if 'V' in numeric_picture else '.'",
        '            integer_digits, fractional_digits = numeric_picture.split(separator, 1)',
        '            text = raw.strip()',
        '            if not text:',
        '                return raw',
        "            if text.count(',') == 1 and '.' not in text:",
        "                text = text.replace(',', '.')",
        "            sign = '-' if text.startswith('-') else ''",
        "            if text[:1] in '+-':",
        '                text = text[1:]',
        "            whole_part, dot, frac_part = text.partition('.')",
        "            whole_digits = ''.join(ch for ch in whole_part if ch.isdigit())",
        "            frac_digits = ''.join(ch for ch in frac_part if ch.isdigit())",
        '            if not whole_digits and not frac_digits:',
        '                return raw',
        '            if len(whole_digits) > len(integer_digits):',
        '                whole_digits = whole_digits[-len(integer_digits):] if integer_digits else ""',
        '                frac_digits = ""',
        '            else:',
        '                frac_digits = frac_digits[:len(fractional_digits)]',
        '            whole_digits = whole_digits or "0"',
        '            frac_digits = frac_digits.ljust(len(fractional_digits), "0")',
        "            return sign + whole_digits + ('.' + frac_digits if fractional_digits else '')",
        "        if expanded.startswith(('X', 'A')):",
        '            return raw[:len(expanded)]',
        '        return raw',
        '',
        '    def _is_numeric(self, value) -> bool:',
        '        if isinstance(value, (int, float, Decimal)):',
        '            return True',
        '        text = str(value).strip()',
        "        if not text:",
        '            return False',
        "        if text[0] in '+-':",
        '            text = text[1:]',
        "        return text.replace('.', '', 1).isdigit()",
        '',
        '    def _is_alphabetic(self, value) -> bool:',
        "        text = str(value).strip()",
        "        return bool(text) and text.isalpha()",
        '',
        '    def _array_value(self, name: str, index):',
        '        values = getattr(self, name, [])',
        "        if not isinstance(values, list) or not values:",
        '            return 0',
        '        try:',
        '            idx = int(index)',
        '        except (TypeError, ValueError):',
        '            idx = 1',
        '        if idx < 1 or idx > len(values):',
        '            return 0',
        '        return values[idx - 1]',
        '',
        '    def _expand_pic(self, pic: str) -> str:',
        "        expanded = (pic or '').upper().replace(' ', '')",
        "        while re.search(r'([X9ZAS])\\((\\d+)\\)', expanded):",
        "            expanded = re.sub(",
        "                r'([X9ZAS])\\((\\d+)\\)',",
        "                lambda m: m.group(1) * int(m.group(2)),",
        '                expanded,',
        '            )',
        '        return expanded',
        '',
        '    def _format_edited_picture(self, pic: str, value) -> str:',
        '        expanded = self._expand_pic(pic)',
        '        try:',
        '            decimal_value = Decimal(str(value).replace(",", "."))',
        '        except Exception:',
            '            return str(value)',
        "        sign = '-' if decimal_value < 0 else ''",
        '        decimal_value = abs(decimal_value)',
        "        decimal_char = self._decimal_separator if self._decimal_separator in expanded else ''",
        '        if decimal_char:',
        '            left_pattern, right_pattern = expanded.rsplit(decimal_char, 1)',
        '        else:',
        "            left_pattern, right_pattern = expanded, ''",
        "        left_slots = [ch for ch in left_pattern if ch in {'9', 'Z'}]",
        "        right_slots = [ch for ch in right_pattern if ch in {'9', 'Z'}]",
        '        scale = len(right_slots)',
        '        width = len(left_slots) + scale',
        '        scaled = int((decimal_value * (10 ** scale)).to_integral_value(rounding=ROUND_DOWN))',
        '        if width > 0:',
        '            scaled %= 10 ** width',
        '        whole_value = scaled // (10 ** scale) if scale else scaled',
        '        frac_value = scaled % (10 ** scale) if scale else 0',
        '        whole_digits = str(whole_value).zfill(len(left_slots))[-len(left_slots):]',
        '        frac_digits = str(frac_value).zfill(scale)',
        '        first_nonzero = next((idx for idx, digit in enumerate(whole_digits) if digit != "0"), None)',
        '        if first_nonzero is None:',
        '            first_visible = next((idx for idx, slot in enumerate(left_slots) if slot == "9"), len(left_slots))',
        '        else:',
        '            first_visible = first_nonzero',
        '        rendered_left: list[str] = []',
        '        digit_index = 0',
        '        visible_seen = False',
        '        for ch in left_pattern:',
        "            if ch in {'9', 'Z'}:",
        '                digit = whole_digits[digit_index] if digit_index < len(whole_digits) else "0"',
        '                if ch == "9" or digit_index >= first_visible:',
        '                    rendered_left.append(digit)',
        '                    if digit != "0" or ch == "9":',
        '                        visible_seen = True',
        '                else:',
        '                    rendered_left.append(" ")',
        '                digit_index += 1',
        '                continue',
        '            if ch == "-":',
        '                rendered_left.append("-" if sign else " ")',
        '                continue',
        '            if ch == "+":',
        '                rendered_left.append(sign or "+")',
        '                continue',
        '            if ch == "$":',
        '                rendered_left.append("$" if visible_seen else " ")',
        '                continue',
        '            if ch in {".", ",", "/"}:',
        '                rendered_left.append(ch if visible_seen else " ")',
        '                continue',
        '            rendered_left.append(ch)',
        '        rendered_right: list[str] = []',
        '        digit_index = 0',
        '        frac_digits = frac_digits[:scale].ljust(scale, "0")',
        '        for ch in right_pattern:',
        "            if ch in {'9', 'Z'}:",
        '                rendered_right.append(frac_digits[digit_index] if digit_index < len(frac_digits) else "0")',
        '                digit_index += 1',
        '                continue',
        '            rendered_right.append(ch)',
        '        rendered = "".join(rendered_left)',
        '        if decimal_char:',
        '            rendered += decimal_char + "".join(rendered_right)',
        '        prefix_sign = sign if "-" not in expanded and "+" not in expanded else ""',
        '        return prefix_sign + rendered',
        '',
        '    def _format_display(self, name: str, value):',
        '        if name in self._pointer_fields:',
        '            if not value:',
        "                return '0x0000000000000000'",
        "            if isinstance(value, dict):",
        "                return value.get('address', '0x0000000000000000')",
        '            return value',
        '        if isinstance(value, list):',
        "            return ''.join(str(self._format_display(name, item)) for item in value)",
        '        if name in self._group_children:',
        "            return ''.join(str(self._format_display(child, getattr(self, child, ''))) for child in self._group_children[name])",
        "        pic = self._field_pics.get(name)",
        '        if not pic:',
        '            return value',
        '        expanded = self._expand_pic(pic)',
        "        if expanded == 'POINTER':",
        '            return value',
        "        if any(token in expanded for token in ('Z', '$', '*', '/', 'CR', 'DB')):",
        '            return self._format_edited_picture(pic, value)',
        "        numeric_picture = expanded.lstrip('S')",
        "        if numeric_picture and set(numeric_picture) <= {'9'}:",
        '            try:',
        '                number = int(value)',
        '            except (TypeError, ValueError):',
        '                return value',
        "            if expanded.startswith('S'):",
        "                sign = '-' if number < 0 else '+'",
        '            else:',
        "                sign = '-' if number < 0 else ''",
        "            return sign + str(abs(number)).zfill(len(numeric_picture))",
        "        if 'V' in numeric_picture or '.' in numeric_picture:",
        "            separator = 'V' if 'V' in numeric_picture else '.'",
        "            integer_digits, fractional_digits = numeric_picture.split(separator, 1)",
        '            try:',
        '                decimal_value = self._coerce_decimal_to_pic(Decimal(str(value).replace(",", ".")), pic)',
        '            except Exception:',
        '                return value',
        "            quantized = f'{abs(decimal_value):0{len(integer_digits) + len(fractional_digits) + 1}.{len(fractional_digits)}f}'",
        "            digits = quantized.replace('.', '')",
        "            sign = '-' if decimal_value < 0 else ''",
        '            whole = digits[:len(integer_digits)].zfill(len(integer_digits))',
        '            frac = digits[-len(fractional_digits):].zfill(len(fractional_digits))',
        '            return sign + whole + self._decimal_separator + frac',
        '        return value',
        '',
        '    def _assign_group_value(self, name: str, value):',
        "        field_specs = self._group_field_specs.get(name, [])",
        '        if not field_specs:',
        '            setattr(self, name, value)',
        '            return',
        "        text = '' if value is None else str(value)",
        '        offset = 0',
        '        for field_spec in field_specs:',
        "            child_name = str(field_spec['name'])",
        "            length = int(field_spec['length'])",
        '            chunk = text[offset:offset + length]',
        "            field_type = self._field_types.get(child_name, 'Any')",
        "            coerced = chunk if field_type == 'str' else (chunk.strip() or '0')",
        "            index = field_spec.get('index')",
        '            if index is None:',
        '                setattr(self, child_name, self._coerce_value(child_name, coerced))',
        '            else:',
        '                existing = list(getattr(self, child_name, []))',
        '                default_item = self._coerce_value(child_name, "" if field_type == "str" else 0)',
        '                while len(existing) <= int(index):',
        '                    existing.append(default_item)',
        '                existing[int(index)] = self._coerce_value(child_name, coerced)',
        '                setattr(self, child_name, existing)',
        '            offset += length',
        '        setattr(self, name, text)',
        '',
        '    def _string_fragment(self, value, delimiter):',
        "        text = '' if value is None else str(value)",
        "        delim_text = '' if delimiter is None else str(delimiter)",
        '        upper = delim_text.upper()',
        "        if upper in ('', 'SIZE'):",
        '            return text',
        "        if upper in ('SPACE', 'SPACES'):",
        "            return text.split(' ', 1)[0]",
        '        head, sep, _ = text.partition(delim_text)',
        '        return head if sep else text',
        '',
        '    def _display_value(self, name: str):',
        "        return self._format_display(name, getattr(self, name, None))",
        '',
        '    def _make_address(self, group_name: str) -> str:',
        "        value = abs(hash((self.__class__.__name__, group_name))) & ((1 << 48) - 1)",
        '        if value == 0:',
        '            value = 0x1000',
        "        return f'0x{value:016x}'",
        '',
        '    def _set_pointer(self, pointer_name: str, group_name: str):',
        "        setattr(self, pointer_name, {'group': group_name, 'address': self._make_address(group_name)})",
        '',
        '    def _assign_address(self, target_group: str, pointer_name: str):',
        '        pointer = getattr(self, pointer_name, None)',
        "        if not isinstance(pointer, dict):",
        '            return',
        "        source_group = pointer.get('group')",
        '        source_children = self._group_children.get(source_group, [])',
        '        target_children = self._group_children.get(target_group, [])',
        '        for src_name, dst_name in zip(source_children, target_children):',
        "            setattr(self, dst_name, getattr(self, src_name, ''))",
        '',
        '    def _resolve_file_path(self, file_name: str) -> Path:',
        "        spec = self._file_specs.get(file_name, {})",
        "        assign_to = str(spec.get('assign_to', '')).strip()",
        "        base_dir = Path(spec.get('base_dir', '.'))",
        '        path = Path(assign_to)',
        '        if not path.is_absolute():',
        '            path = (base_dir / path).resolve()',
        '        return path',
        '',
        '    def _open_input(self, file_name: str):',
        "        spec = self._file_specs.get(file_name, {})",
        "        status_var = str(spec.get('status_var', ''))",
        '        path = self._resolve_file_path(file_name)',
        '        if path.exists():',
        "            with path.open('r', encoding='utf-8', errors='replace') as handle:",
        "                records = [line.rstrip('\\r\\n') for line in handle]",
        "            self._file_handles[file_name] = {'records': records, 'index': 0}",
        "            if status_var: setattr(self, status_var, '00')",
        '            return',
        "        self._file_handles[file_name] = {'records': [], 'index': 0}",
        "        if status_var: setattr(self, status_var, '35')",
        '',
        '    def _close_file(self, file_name: str):',
        "        if file_name in self._file_handles:",
        '            del self._file_handles[file_name]',
        '',
        '    def _read_record(self, file_name: str) -> bool:',
        "        handle = self._file_handles.get(file_name)",
        "        if not handle:",
        '            return False',
        "        records = handle.get('records', [])",
        "        index = int(handle.get('index', 0))",
        '        if index >= len(records):',
        '            return False',
        '        record = records[index]',
        "        handle['index'] = index + 1",
        "        spec = self._file_specs.get(file_name, {})",
        "        for field_spec in spec.get('fields', []):",
        "            name = str(field_spec['name'])",
        "            start = int(field_spec['start'])",
        "            length = int(field_spec['length'])",
        '            chunk = record[start:start + length]',
        "            field_type = self._field_types.get(name, 'Any')",
        "            value = chunk if field_type == 'str' else chunk.strip()",
        '            setattr(self, name, self._coerce_value(name, value if value != "" else 0))',
        '        return True',
        '',
        '    def _perform_through(self, start: str, end: str):',
        '        started = False',
        '        for paragraph in self._paragraph_order:',
        '            if paragraph == start:',
        '                started = True',
        '            if started:',
        '                getattr(self, paragraph)()',
        '            if started and paragraph == end:',
        '                break',
    ])
    return '\n'.join(lines)


def _translate_move(stmt: Statement) -> tuple[str, bool]:
    """MOVE x TO y  ->  y = x"""
    raw = stmt.raw
    m = re.match(r'MOVE\s+(.+?)\s+TO\s+(.+)', raw, re.IGNORECASE | re.DOTALL)
    if not m:
        return f'# UNHANDLED: {stmt.raw}', False
    src, dst = m.group(1).strip(), m.group(2).strip().rstrip('.')
    py_src = _cobol_expr_to_python(src)
    py_dst = _cobol_target_to_python(dst)
    field_name = _target_field_name(dst)
    if _CURRENT_FIELD_TYPES.get(field_name) == 'str':
        target_pic = (_CURRENT_FIELD_PICS.get(field_name) or '').upper().replace(' ', '').rstrip('.')
        if any(token in target_pic for token in ('Z', '$', '*', '/', 'CR', 'DB')):
            py_src = f'self._format_display("{field_name}", {_cobol_expr_to_python(src)})'
        else:
            py_src = _simple_source_display_expr(src) or py_src
    if field_name.upper() in _CURRENT_GROUP_FIELDS:
        return f'self._assign_group_value("{field_name}", {py_src})', True
    return f'{py_dst} = self._coerce_value("{field_name}", {py_src})', True


def _translate_compute(stmt: Statement) -> tuple[str, bool]:
    """COMPUTE y = expr  ->  y = expr (with operator substitution)"""
    raw = stmt.raw
    m = re.match(r'COMPUTE\s+(\S+)\s*=\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
    if not m:
        return f'# UNHANDLED: {stmt.raw}', False
    raw_dst = m.group(1).strip()
    dst  = _cobol_target_to_python(raw_dst)
    expr = _cobol_expr_to_python(m.group(2).strip())
    field_name = _target_field_name(raw_dst)
    return f'{dst} = self._coerce_value("{field_name}", {expr})', True


def _translate_arithmetic(stmt: Statement) -> tuple[str, bool]:
    """ADD/SUBTRACT/MULTIPLY/DIVIDE x TO/FROM/BY/INTO y"""
    v = stmt.verb.upper()
    raw = ' '.join(stmt.raw.strip().rstrip('.').split())
    m = None
    op = ''
    if v == 'ADD':
        giving = re.match(r'ADD\s+(.+?)\s+TO\s+(\S+)\s+GIVING\s+(\S+)$', raw, re.IGNORECASE)
        if giving:
            src_expr, other, raw_dst = giving.groups()
            dst = _cobol_target_to_python(raw_dst)
            field_name = _target_field_name(raw_dst)
            terms = [_cobol_expr_to_python(term) for term in src_expr.split()]
            total_expr = ' + '.join([_cobol_expr_to_python(other), *terms])
            return (
                f'{dst} = self._coerce_value("{field_name}", '
                f'{total_expr})',
                True,
            )
        m = re.match(r'ADD\s+(.+?)\s+TO\s+(\S+)$', raw, re.IGNORECASE)
        if m:
            src_expr, raw_dst = m.groups()
            dst = _cobol_target_to_python(raw_dst)
            field_name = _target_field_name(raw_dst)
            terms = [_cobol_expr_to_python(term) for term in src_expr.split()]
            total_expr = ' + '.join([dst, *terms])
            return f'{dst} = self._coerce_value("{field_name}", {total_expr})', True
    elif v == 'SUBTRACT':
        giving = re.match(r'SUBTRACT\s+(.+?)\s+FROM\s+(\S+)\s+GIVING\s+(\S+)$', raw, re.IGNORECASE)
        if giving:
            src_expr, other, raw_dst = giving.groups()
            dst = _cobol_target_to_python(raw_dst)
            field_name = _target_field_name(raw_dst)
            terms = [_cobol_expr_to_python(term) for term in src_expr.split()]
            total_expr = ' - '.join([_cobol_expr_to_python(other), *terms])
            return (
                f'{dst} = self._coerce_value("{field_name}", '
                f'{total_expr})',
                True,
            )
        m = re.match(r'SUBTRACT\s+(.+?)\s+FROM\s+(\S+)$', raw, re.IGNORECASE)
        if m:
            src_expr, raw_dst = m.groups()
            dst = _cobol_target_to_python(raw_dst)
            field_name = _target_field_name(raw_dst)
            terms = [_cobol_expr_to_python(term) for term in src_expr.split()]
            total_expr = ' - '.join([dst, *terms])
            return f'{dst} = self._coerce_value("{field_name}", {total_expr})', True
    elif v == 'MULTIPLY':
        giving = re.match(r'MULTIPLY\s+(\S+)\s+BY\s+(\S+)\s+GIVING\s+(\S+)', raw, re.IGNORECASE)
        if giving:
            src, other, raw_dst = giving.groups()
            dst = _cobol_target_to_python(raw_dst)
            field_name = _target_field_name(raw_dst)
            return (
                f'{dst} = self._coerce_value("{field_name}", '
                f'{_cobol_expr_to_python(other)} * {_cobol_expr_to_python(src)})',
                True,
            )
        m = re.match(r'MULTIPLY\s+(\S+)\s+BY\s+(\S+)', raw, re.IGNORECASE)
        op = '*='
    elif v == 'DIVIDE':
        giving_remainder = re.match(
            r'DIVIDE\s+(\S+)\s+BY\s+(\S+)\s+GIVING\s+(\S+)\s+REMAINDER\s+(\S+)$',
            raw,
            re.IGNORECASE,
        )
        if giving_remainder:
            dividend, divisor, raw_dst, raw_rem = giving_remainder.groups()
            dst = _cobol_target_to_python(raw_dst)
            rem = _cobol_target_to_python(raw_rem)
            dst_name = _target_field_name(raw_dst)
            rem_name = _target_field_name(raw_rem)
            py_dividend = _cobol_expr_to_python(dividend)
            py_divisor = _cobol_expr_to_python(divisor)
            return (
                '\n'.join([
                    f'if {py_divisor} == 0:',
                    f'    {dst} = self._coerce_value("{dst_name}", 0)',
                    f'    {rem} = self._coerce_value("{rem_name}", 0)',
                    'else:',
                    f'    {dst} = self._coerce_value("{dst_name}", int({py_dividend} / {py_divisor}))',
                    f'    {rem} = self._coerce_value("{rem_name}", int({py_dividend} % {py_divisor}))',
                ]),
                True,
            )
        giving_by = re.match(r'DIVIDE\s+(\S+)\s+BY\s+(\S+)\s+GIVING\s+(\S+)$', raw, re.IGNORECASE)
        if giving_by:
            dividend, divisor, raw_dst = giving_by.groups()
            dst = _cobol_target_to_python(raw_dst)
            field_name = _target_field_name(raw_dst)
            return (
                f'{dst} = self._coerce_value("{field_name}", '
                f'0 if {_cobol_expr_to_python(divisor)} == 0 else {_cobol_expr_to_python(dividend)} / {_cobol_expr_to_python(divisor)})',
                True,
            )
        giving = re.match(r'DIVIDE\s+(\S+)\s+INTO\s+(\S+)\s+GIVING\s+(\S+)', raw, re.IGNORECASE)
        if giving:
            src, other, raw_dst = giving.groups()
            dst = _cobol_target_to_python(raw_dst)
            field_name = _target_field_name(raw_dst)
            return (
                f'{dst} = self._coerce_value("{field_name}", '
                f'{_cobol_expr_to_python(other)} / {_cobol_expr_to_python(src)})',
                True,
            )
        m = re.match(r'DIVIDE\s+(\S+)\s+INTO\s+(\S+)', raw, re.IGNORECASE)
        op = '/='
    else:
        return f'# UNHANDLED: {stmt.raw}', False
    if not m:
        return f'# UNHANDLED: {stmt.raw}', False
    src = _cobol_expr_to_python(m.group(1))
    raw_dst = m.group(2)
    dst = _cobol_target_to_python(raw_dst)
    field_name = _target_field_name(raw_dst)
    op_symbol = {
        '+=': '+',
        '-=': '-',
        '*=': '*',
        '/=': '/',
    }[op]
    return f'{dst} = self._coerce_value("{field_name}", {dst} {op_symbol} {src})', True


def _statement_from_raw(raw: str) -> Statement:
    """Create a Statement object from raw COBOL text for recursive block translation."""
    first_line = raw.strip().splitlines()[0].strip()
    norm = _normalise_cobol(first_line)
    match = _COBOL_VERB_RE.match(norm)
    verb = match.group(1).upper() if match else 'UNKNOWN'
    args_str = norm[len(match.group(0)):].strip() if match else norm
    return Statement(verb=verb, raw=raw.strip(), args=args_str.split() if args_str else [])


def _is_perform_block_start(norm: str) -> bool:
    return bool(
        re.match(r'^PERFORM\s+UNTIL\b', norm) or
        re.match(r'^PERFORM\s+VARYING\b', norm) or
        re.match(r'^PERFORM\s+.+\s+TIMES(?:\s|$)', norm)
    )


def _is_evaluate_block_start(norm: str) -> bool:
    return norm.startswith('EVALUATE ')


def _is_string_block_start(norm: str) -> bool:
    return norm.startswith('STRING ')


def _is_read_block_start(norm: str) -> bool:
    return norm.startswith('READ ')


def _split_block_items(body: str) -> list[str]:
    """Split a COBOL multi-line block body into nested statement chunks."""
    lines: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        lines.extend(_split_inline_statements(line))

    def _collect_nested_block(
        start_idx: int,
        start_predicate,
        end_token: str,
    ) -> tuple[str, int]:
        depth = 1
        chunk = [lines[start_idx]]
        i = start_idx + 1
        while i < len(lines):
            current = lines[i]
            current_norm = _normalise_cobol(current)
            if start_predicate(current_norm):
                depth += 1
            elif current_norm.startswith(end_token):
                depth -= 1
            chunk.append(current)
            i += 1
            if depth == 0:
                break
        return '\n'.join(chunk), i

    norms = [_normalise_cobol(line) for line in lines]

    def _consume_simple_statement(start_idx: int) -> tuple[str, int]:
        chunk = [lines[start_idx]]
        i = start_idx + 1
        while i < len(lines):
            if _COBOL_VERB_RE.match(norms[i]):
                break
            chunk.append(lines[i])
            i += 1
        return '\n'.join(chunk), i

    def _consume_statement(start_idx: int) -> tuple[str, int]:
        norm = norms[start_idx]
        if _is_perform_block_start(norm):
            has_end_perform = any(candidate.startswith('END-PERFORM') for candidate in norms[start_idx + 1:])
            if has_end_perform:
                return _collect_nested_block(start_idx, _is_perform_block_start, 'END-PERFORM')
        if _is_evaluate_block_start(norm):
            return _collect_nested_block(start_idx, _is_evaluate_block_start, 'END-EVALUATE')
        if _is_string_block_start(norm):
            has_end_string = any(candidate.startswith('END-STRING') for candidate in norms[start_idx + 1:])
            if has_end_string:
                return _collect_nested_block(start_idx, _is_string_block_start, 'END-STRING')
        if _is_read_block_start(norm):
            has_end_read = any(candidate.startswith('END-READ') for candidate in norms[start_idx + 1:])
            if has_end_read:
                return _collect_nested_block(start_idx, _is_read_block_start, 'END-READ')
        if norm.startswith('IF'):
            return _consume_if(start_idx)
        return _consume_simple_statement(start_idx)

    def _consume_if(start_idx: int) -> tuple[str, int]:
        chunk = [lines[start_idx]]
        i = start_idx + 1
        consumed_true_branch = False
        while i < len(lines):
            norm = norms[i]
            if norm.startswith('ELSE') or norm.startswith('END-IF'):
                break
            stmt_chunk, i = _consume_statement(i)
            chunk.extend(stmt_chunk.splitlines())
            if not consumed_true_branch:
                consumed_true_branch = True
                if i >= len(lines):
                    return '\n'.join(chunk), i
                next_norm = norms[i]
                if not (next_norm.startswith('ELSE') or next_norm.startswith('END-IF')):
                    return '\n'.join(chunk), i

        if i < len(lines) and norms[i].startswith('ELSE'):
            chunk.append(lines[i])
            i += 1
            consumed_false_branch = False
            while i < len(lines):
                if norms[i].startswith('END-IF'):
                    break
                stmt_chunk, i = _consume_statement(i)
                chunk.extend(stmt_chunk.splitlines())
                if not consumed_false_branch:
                    consumed_false_branch = True
                    if i >= len(lines):
                        return '\n'.join(chunk), i
                    if not norms[i].startswith('END-IF'):
                        return '\n'.join(chunk), i

        if i < len(lines) and norms[i].startswith('END-IF'):
            chunk.append(lines[i])
            i += 1
        return '\n'.join(chunk), i

    items: list[str] = []
    i = 0
    while i < len(lines):
        chunk, i = _consume_statement(i)
        items.append(chunk)
    return items


def _translate_body_block(body: str, indent: int = 4) -> str:
    """Recursively translate a COBOL statement block into Python source lines."""
    translated: list[str] = []
    for raw_stmt in _split_block_items(body):
        py_line, _ = translate_statement(_statement_from_raw(raw_stmt))
        lines = py_line.splitlines() if py_line else ['pass']
        if not lines:
            lines = ['pass']
        for line in lines:
            translated.append((' ' * indent) + line)
    if not translated:
        translated.append((' ' * indent) + 'pass')
    return '\n'.join(translated)


def _translate_if(stmt: Statement) -> tuple[str, bool]:
    """IF cond ... -> if cond: (body translated separately)"""
    raw = stmt.raw.strip().rstrip('.')
    if '\n' in raw:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        header = lines[0]
        cond = re.sub(r'^IF\s*', '', header, flags=re.IGNORECASE).strip()
        cond = _cobol_cond_to_python(cond)
        body_lines = lines[1:]
        if body_lines and _normalise_cobol(body_lines[-1]).startswith('END-IF'):
            nested_depth = 0
            for candidate in body_lines[:-1]:
                norm = _normalise_cobol(candidate)
                if norm.startswith('IF'):
                    nested_depth += 1
                elif norm.startswith('END-IF'):
                    nested_depth = max(0, nested_depth - 1)
            if nested_depth == 0:
                body_lines = body_lines[:-1]
        else_index = None
        nested_if = 0
        for i, line in enumerate(body_lines):
            norm = _normalise_cobol(line)
            if norm.startswith('IF'):
                nested_if += 1
            elif norm.startswith('END-IF'):
                nested_if = max(0, nested_if - 1)
            elif norm.startswith('ELSE') and nested_if == 0:
                else_index = i
                break
        true_body = body_lines if else_index is None else body_lines[:else_index]
        false_body = [] if else_index is None else body_lines[else_index + 1:]
        block = [f'if {cond}:', _translate_body_block('\n'.join(true_body))]
        if else_index is not None:
            block.append('else:')
            block.append(_translate_body_block('\n'.join(false_body)))
        return '\n'.join(block), True
    m = re.match(r'IF\s+(.+)', raw, re.IGNORECASE)
    if not m:
        return f'# UNHANDLED IF: {stmt.raw}', False
    cond = m.group(1).strip()
    cond = _cobol_cond_to_python(cond)
    return f'if {cond}:', True


def _translate_display(stmt: Statement) -> tuple[str, bool]:
    """DISPLAY x y z -> print(f'...')"""
    raw = stmt.raw.strip().rstrip('.')
    m = re.match(r'DISPLAY\s+(.+)', raw, re.IGNORECASE | re.DOTALL)
    if not m:
        return f'# UNHANDLED: {stmt.raw}', False
    args_str = _replace_qualified_names(m.group(1).strip())

    def _collapse_literal_continuations(text: str) -> str:
        pieces: list[str] = []
        active_quote: str | None = None
        i = 0
        while i < len(text):
            ch = text[i]
            if active_quote is not None:
                if ch in '\r\n':
                    i += 1
                    while i < len(text) and text[i] in ' \t-':
                        i += 1
                    if i < len(text) and text[i] == active_quote:
                        i += 1
                    continue
                pieces.append(ch)
                if ch == active_quote:
                    active_quote = None
                i += 1
                continue
            if ch in "'\"":
                active_quote = ch
                pieces.append(ch)
                i += 1
                continue
            pieces.append(' ' if ch in '\r\n' else ch)
            i += 1
        return ''.join(pieces)

    args_str = _collapse_literal_continuations(args_str)
    tokens = re.findall(
        r"'[^']*'|\"[^\"]*\"|[A-Z][A-Z0-9\-_]*(?:\([^)\n]+\))?|[+-]?\d+(?:[.,]\d+)?",
        args_str,
        flags=re.IGNORECASE | re.DOTALL,
    )
    parts = []
    for tok in tokens:
        if tok.startswith(("'", '"')):
            literal = tok[1:-1]
            literal = re.sub(r'\s*\n\s*', '', literal)
            parts.append(literal)
        elif re.fullmatch(r'([A-Z][A-Z0-9\-_]*)\([^)\n]+\)', tok, flags=re.IGNORECASE):
            base_name = _cobol_name_to_python(tok.split('(', 1)[0])
            parts.append('{' + f'self._format_display("{base_name}", {_cobol_expr_to_python(tok)})' + '}')
        elif re.fullmatch(r'[A-Z][A-Z0-9\-]*', tok, flags=re.IGNORECASE):
            parts.append('{self._display_value("' + _cobol_name_to_python(tok) + '")}')
        else:
            parts.append('{' + _cobol_expr_to_python(tok) + '}')
    return f"print(f'{''.join(parts)}')", True


def _translate_accept(stmt: Statement) -> tuple[str, bool]:
    """Translate ACCEPT target [FROM TIME|DATE]."""
    raw = stmt.raw.strip().rstrip('.')
    time_match = re.match(r'ACCEPT\s+(.+?)\s+FROM\s+TIME\b', raw, re.IGNORECASE)
    if time_match:
        raw_target = time_match.group(1).strip()
        field_name = _target_field_name(raw_target)
        value_expr = 'time.strftime("%H%M%S") + f"{int((time.time() % 1) * 100):02d}"'
        if field_name.upper() in _CURRENT_GROUP_FIELDS:
            return f'self._assign_group_value("{field_name}", {value_expr})', True
        target = _cobol_target_to_python(raw_target)
        return f'{target} = self._coerce_value("{field_name}", {value_expr})', True
    date_match = re.match(r'ACCEPT\s+(.+?)\s+FROM\s+DATE(?:\s+YYYYMMDD)?\b', raw, re.IGNORECASE)
    if date_match:
        raw_target = date_match.group(1).strip()
        field_name = _target_field_name(raw_target)
        value_expr = 'time.strftime("%Y%m%d")'
        if field_name.upper() in _CURRENT_GROUP_FIELDS:
            return f'self._assign_group_value("{field_name}", {value_expr})', True
        target = _cobol_target_to_python(raw_target)
        return f'{target} = self._coerce_value("{field_name}", {value_expr})', True
    args = ' '.join(stmt.args)
    raw_target = args.split(' FROM ', 1)[0] if args else 'input_val'
    py_var = _cobol_target_to_python(raw_target)
    field_name = _target_field_name(raw_target)
    if field_name.upper() in _CURRENT_GROUP_FIELDS:
        return f'self._assign_group_value("{field_name}", self._accept_input("{field_name}"))', True
    return f'{py_var} = self._coerce_value("{field_name}", self._accept_input("{field_name}"))', True


def _translate_string(stmt: Statement) -> tuple[str, bool]:
    """Translate a basic STRING ... INTO ... END-STRING block."""
    raw = stmt.raw.strip().rstrip('.')
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    flat = ' '.join(lines)
    string_body = re.sub(r'^STRING\s+', '', flat, flags=re.IGNORECASE)
    into_marker = re.search(r'\bINTO\b', string_body, flags=re.IGNORECASE)
    if not into_marker:
        return f'# UNHANDLED STRING: {stmt.raw}', False
    pieces_text = string_body[:into_marker.start()].strip()
    target_match = re.search(
        r'\bINTO\s+([A-Z][A-Z0-9\-_]*(?:\([^)\n]+\))?(?:\s+OF\s+[A-Z][A-Z0-9\-_]*)?)',
        flat,
        flags=re.IGNORECASE,
    )
    piece_matches = list(re.finditer(
        r'("[^"]*"|\'[^\']*\'|[A-Z][A-Z0-9\-_]*(?:\([^)\n]+\))?(?:\s+OF\s+[A-Z][A-Z0-9\-_]*)?)\s+DELIMITED\s+BY\s+(SIZE|SPACE|SPACES|"[^"]*"|\'[^\']*\'|[A-Z][A-Z0-9\-_]*)',
        pieces_text,
        flags=re.IGNORECASE,
    ))
    if not piece_matches or not target_match:
        return f'# UNHANDLED STRING: {stmt.raw}', False
    raw_target = target_match.group(1)
    target = _cobol_target_to_python(raw_target)
    field_name = _target_field_name(raw_target)
    fragments: list[str] = []
    for match in piece_matches:
        value_token, delimiter_token = match.groups()
        if value_token.startswith(("'", '"')):
            value_expr = repr(value_token.strip("'\""))
        else:
            value_expr = _cobol_expr_to_python(value_token)
        if delimiter_token.startswith(("'", '"')):
            delimiter_expr = repr(delimiter_token.strip("'\""))
        else:
            delimiter_expr = repr(delimiter_token.upper())
        fragments.append(f'self._string_fragment({value_expr}, {delimiter_expr})')
    joined = ' + '.join(fragments)
    return f'{target} = self._coerce_value("{field_name}", {joined})', True


def _translate_perform(stmt: Statement) -> tuple[str, bool]:
    """PERFORM para -> self.para() or # TODO: complex PERFORM"""
    raw = stmt.raw.strip().rstrip('.')
    m = re.match(r'PERFORM\s+(\S+)\s+(?:THROUGH|THRU)\s+(\S+)\s*$', raw, re.IGNORECASE)
    if m:
        start_para = _cobol_name_to_python(m.group(1))
        end_para = _cobol_name_to_python(m.group(2))
        return f"self._perform_through('{start_para}', '{end_para}')", True
    m = re.match(r'PERFORM\s+(\S+)\s+UNTIL\s+(.+)$', raw, re.IGNORECASE)
    if m:
        para = _cobol_name_to_python(m.group(1))
        py_cond = _cobol_cond_to_python(m.group(2))
        return '\n'.join([
            f'while not ({py_cond}):',
            f'    self.{para}()',
        ]), True
    # Simple PERFORM para
    m = re.match(r'PERFORM\s+(\S+)\s*$', raw, re.IGNORECASE)
    if m:
        para = _cobol_name_to_python(m.group(1))
        return f'self.{para}()', True
    if '\n' in raw:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        header = lines[0]
        body_lines = lines[1:]
        if body_lines and _normalise_cobol(body_lines[-1]).startswith('END-PERFORM'):
            body_lines = body_lines[:-1]
        body = '\n'.join(body_lines)
        varying_continued = re.match(
            r'PERFORM\s+VARYING\s+(\S+)\s+FROM\s+(.+?)\s+BY\s+(.+?)\s*$',
            header,
            re.IGNORECASE,
        )
        if varying_continued and body_lines and _normalise_cobol(body_lines[0]).startswith('UNTIL '):
            var, start, step = varying_continued.groups()
            cond = re.sub(r'^UNTIL\s+', '', body_lines[0], flags=re.IGNORECASE).strip()
            py_var = _cobol_target_to_python(var)
            py_start = _cobol_expr_to_python(start)
            py_step = _cobol_expr_to_python(step)
            py_cond = _cobol_cond_to_python(cond)
            block_body = '\n'.join(body_lines[1:])
            block = [
                f'{py_var} = {py_start}',
                f'while not ({py_cond}):',
                _translate_body_block(block_body, indent=4),
                f'    {py_var} += {py_step}',
            ]
            return '\n'.join(block), True
        varying = re.match(
            r'PERFORM\s+VARYING\s+(\S+)\s+FROM\s+(.+?)\s+BY\s+(.+?)\s+UNTIL\s+(.+)',
            header,
            re.IGNORECASE,
        )
        if varying:
            var, start, step, cond = varying.groups()
            py_var = _cobol_target_to_python(var)
            py_start = _cobol_expr_to_python(start)
            py_step = _cobol_expr_to_python(step)
            py_cond = _cobol_cond_to_python(cond)
            block = [
                f'{py_var} = {py_start}',
                f'while not ({py_cond}):',
                _translate_body_block(body, indent=4),
                f'    {py_var} += {py_step}',
            ]
            return '\n'.join(block), True
        until = re.match(r'PERFORM\s+UNTIL\s+(.+)', header, re.IGNORECASE)
        if until:
            py_cond = _cobol_cond_to_python(until.group(1))
            return '\n'.join([
                f'while not ({py_cond}):',
                _translate_body_block(body, indent=4),
            ]), True
        times = re.match(r'PERFORM\s+(.+?)\s+TIMES', header, re.IGNORECASE)
        if times:
            count_expr = _cobol_expr_to_python(times.group(1))
            return '\n'.join([
                f'for _ in range(int({count_expr})):',
                _translate_body_block(body, indent=4),
            ]), True
    # PERFORM VARYING / UNTIL -- leave to LLM when still too complex
    if re.search(r'VARYING|UNTIL|TIMES', raw, re.IGNORECASE):
        return f'# TODO [OLLAMA]: {stmt.raw}', False
    return f'# UNHANDLED PERFORM: {stmt.raw}', False


def _translate_open(stmt: Statement) -> tuple[str, bool]:
    """Translate OPEN INPUT file-name into the runtime helper."""
    raw = stmt.raw.strip().rstrip('.')
    m = re.match(r'OPEN\s+INPUT\s+(\S+)', raw, re.IGNORECASE)
    if not m:
        return f'# TODO [OLLAMA]: {stmt.raw}', False
    return f"self._open_input('{_cobol_name_to_python(m.group(1))}')", True


def _translate_close(stmt: Statement) -> tuple[str, bool]:
    """Translate CLOSE file-name into the runtime helper."""
    raw = stmt.raw.strip().rstrip('.')
    m = re.match(r'CLOSE\s+(\S+)', raw, re.IGNORECASE)
    if not m:
        return f'# TODO [OLLAMA]: {stmt.raw}', False
    return f"self._close_file('{_cobol_name_to_python(m.group(1))}')", True


def _translate_read(stmt: Statement) -> tuple[str, bool]:
    """Translate READ ... AT END / NOT AT END blocks for sequential input."""
    raw = stmt.raw.strip().rstrip('.')
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return f'# TODO [OLLAMA]: {stmt.raw}', False
    header = lines[0]
    m = re.match(r'READ\s+(\S+)(?:\s+NEXT)?', header, re.IGNORECASE)
    if not m:
        return f'# TODO [OLLAMA]: {stmt.raw}', False
    file_name = _cobol_name_to_python(m.group(1))
    at_end_lines: list[str] = []
    not_at_end_lines: list[str] = []
    target = at_end_lines
    for line in lines[1:]:
        norm = _normalise_cobol(line)
        if norm.startswith('END-READ'):
            break
        if norm.startswith('AT END'):
            target = at_end_lines
            remainder = re.sub(r'^AT\s+END\b', '', line, flags=re.IGNORECASE).strip()
            if remainder:
                target.append(remainder)
            continue
        if norm.startswith('NOT AT END'):
            target = not_at_end_lines
            remainder = re.sub(r'^NOT\s+AT\s+END\b', '', line, flags=re.IGNORECASE).strip()
            if remainder:
                target.append(remainder)
            continue
        target.append(line)
    block = [f"if not self._read_record('{file_name}'):"]
    block.append(_translate_body_block('\n'.join(at_end_lines)))
    if not_at_end_lines:
        block.append('else:')
        block.append(_translate_body_block('\n'.join(not_at_end_lines)))
    return '\n'.join(block), True


def _translate_set(stmt: Statement) -> tuple[str, bool]:
    """Translate common COBOL SET forms, especially pointer/address semantics."""
    raw = stmt.raw.strip().rstrip('.')
    m = re.match(r'SET\s+(\S+)\s+TO\s+TRUE', raw, re.IGNORECASE)
    if m:
        assignment = _CURRENT_CONDITION_ASSIGNMENTS.get(m.group(1).upper())
        if assignment:
            parent_name, literal = assignment
            return (
                f'self.{parent_name} = self._coerce_value("{parent_name}", {literal})',
                True,
            )
    m = re.match(r'SET\s+(\S+)\s+TO\s+ADDRESS\s+OF\s+(\S+)', raw, re.IGNORECASE)
    if m:
        pointer_name = _cobol_name_to_python(m.group(1))
        group_name = _cobol_name_to_python(m.group(2))
        return f"self._set_pointer('{pointer_name}', '{group_name}')", True
    m = re.match(r'SET\s+ADDRESS\s+OF\s+(\S+)\s+TO\s+(\S+)', raw, re.IGNORECASE)
    if m:
        group_name = _cobol_name_to_python(m.group(1))
        pointer_name = _cobol_name_to_python(m.group(2))
        return f"self._assign_address('{group_name}', '{pointer_name}')", True
    m = re.match(r'SET\s+(\S+)\s+TO\s+(.+)', raw, re.IGNORECASE)
    if m:
        target = _cobol_target_to_python(m.group(1))
        value = _cobol_expr_to_python(m.group(2))
        return f'{target} = {value}', True
    return f'# UNHANDLED SET: {stmt.raw}', False


def _translate_stop(stmt: Statement) -> tuple[str, bool]:
    """Translate STOP RUN / STOP literal."""
    raw = stmt.raw.strip().rstrip('.')
    if re.fullmatch(r'(STOP\s+RUN|GOBACK)', raw, re.IGNORECASE):
        return 'sys.exit(0)', True
    message = re.match(r"STOP\s+('([^']*)'|\"([^\"]*)\")$", raw, re.IGNORECASE)
    if message:
        literal = (message.group(2) or message.group(3) or '')
        return '\n'.join([
            f'print({literal!r})',
            'sys.exit(0)',
        ]), True
    if re.fullmatch(r'STOP', raw, re.IGNORECASE):
        return 'sys.exit(0)', True
    return f'# UNHANDLED STOP: {stmt.raw}', False


def _translate_evaluate(stmt: Statement) -> tuple[str, bool]:
    """Translate common EVALUATE forms into if/elif/else chains."""
    raw = stmt.raw.strip().rstrip('.')
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return '# UNHANDLED EVALUATE', False
    header = lines[0]
    subject = re.sub(r'^EVALUATE\s+', '', header, flags=re.IGNORECASE).strip()
    body_lines = lines[1:]
    cases: list[tuple[str, list[str]]] = []
    current_when: str | None = None
    current_body: list[str] = []
    for line in body_lines:
        norm = _normalise_cobol(line)
        if norm.startswith('END-EVALUATE'):
            break
        if norm.startswith('WHEN '):
            if current_when is not None:
                cases.append((current_when, current_body))
            current_when = re.sub(r'^WHEN\s+', '', line, flags=re.IGNORECASE).strip()
            current_body = []
            continue
        current_body.append(line)
    if current_when is not None:
        cases.append((current_when, current_body))
    if not cases:
        return f'# UNHANDLED EVALUATE: {stmt.raw}', False

    subject_is_true = subject.upper() == 'TRUE'
    subject_expr = _cobol_expr_to_python(subject) if not subject_is_true else ''
    block: list[str] = []
    first_branch = True
    for when_cond, when_body in cases:
        norm_when = _normalise_cobol(when_cond)
        if norm_when == 'OTHER':
            block.append('else:')
            block.append(_translate_body_block('\n'.join(when_body)))
            continue
        if subject_is_true:
            py_cond = _cobol_cond_to_python(when_cond)
        else:
            thru = re.match(r'(.+?)\s+THRU(?:OUGH)?\s+(.+)', when_cond, re.IGNORECASE)
            if thru:
                low_expr = _cobol_expr_to_python(thru.group(1).strip())
                high_expr = _cobol_expr_to_python(thru.group(2).strip())
                py_cond = f'({low_expr} <= {subject_expr} <= {high_expr})'
            else:
                py_cond = f'{subject_expr} == {_cobol_expr_to_python(when_cond)}'
        keyword = 'if' if first_branch else 'elif'
        block.append(f'{keyword} {py_cond}:')
        block.append(_translate_body_block('\n'.join(when_body)))
        first_branch = False
    return '\n'.join(block), True


def translate_statement(stmt: Statement) -> tuple[str, bool]:
    """
    Dispatch a COBOL Statement to the appropriate rule translator.
    Returns (python_line, was_handled).
    was_handled=False means the statement is queued for the LLM layer.
    """
    v = stmt.verb.upper()
    if v == 'MOVE':          return _translate_move(stmt)
    if v == 'COMPUTE':       return _translate_compute(stmt)
    if v in ('ADD','SUBTRACT','MULTIPLY','DIVIDE'):
                             return _translate_arithmetic(stmt)
    if v == 'IF':            return _translate_if(stmt)
    if v == 'EVALUATE':      return _translate_evaluate(stmt)
    if v in ('ELSE','END-IF'):
        return ('else:' if v == 'ELSE' else '# end-if', True)
    if v == 'DISPLAY':       return _translate_display(stmt)
    if v == 'ACCEPT':        return _translate_accept(stmt)
    if v == 'PERFORM':       return _translate_perform(stmt)
    if v == 'STRING':        return _translate_string(stmt)
    if v == 'SET':           return _translate_set(stmt)
    if v == 'OPEN':          return _translate_open(stmt)
    if v == 'CLOSE':         return _translate_close(stmt)
    if v == 'READ':          return _translate_read(stmt)
    if v in ('STOP RUN', 'GOBACK', 'STOP'):
        return _translate_stop(stmt)
    if v == 'CALL':
        prog = stmt.args[0].strip("'\"''") if stmt.args else 'UNKNOWN'
        py_prog = _cobol_name_to_python(prog)
        return (f'# TODO [STUB]: {py_prog}_instance.run()  '
                f'# CALL {prog}'), True
    if v in ('CONTINUE', 'NEXT SENTENCE'):
        return 'pass', True
    if v == 'END-STRING':
        return '# end-string', True
    if v == 'END-READ':
        return '# end-read', True
    if v == 'INITIALIZE':
        targets = ' '.join(stmt.args)
        return f'# INITIALIZE {targets}  # TODO: set to defaults', False
    if v == 'EXIT':
        return 'pass', True
    # Unknown verb
    return f'# TODO [OLLAMA]: {stmt.raw}', False


def rule_translate_program(program: COBOLProgram) -> tuple[str, float, list[str]]:
    """
    Run the rule-based engine over an entire COBOLProgram.
    Returns (python_source, rule_coverage_fraction, unhandled_list).
    """
    global _CURRENT_CONDITION_CHECKS, _CURRENT_CONDITION_ASSIGNMENTS
    global _CURRENT_GROUP_FIELDS, _CURRENT_FIELD_TYPES, _CURRENT_FIELD_PICS
    data_section = translate_data_division(program)
    method_lines: list[str] = []
    total_stmts = 0
    handled_stmts = 0
    unhandled: list[str] = []
    _TRANSLATION_CONTEXT_LOCK.acquire()
    _CURRENT_CONDITION_CHECKS = _extract_condition_checks(program)
    _CURRENT_CONDITION_ASSIGNMENTS = _extract_condition_assignments(program)
    _CURRENT_GROUP_FIELDS = {
        item.name.upper()
        for item in _walk_data_items(program.all_data_items)
        if item.children and any(child.level != 88 for child in item.children)
    }
    _CURRENT_GROUP_FIELDS |= {
        _cobol_name_to_python(item.name).upper()
        for item in _walk_data_items(program.all_data_items)
        if item.children and any(child.level != 88 for child in item.children)
    }
    _CURRENT_FIELD_TYPES = {
        _cobol_name_to_python(item.name): _field_base_type(item)
        for item in _walk_data_items(program.all_data_items)
    }
    _CURRENT_FIELD_PICS = {
        _cobol_name_to_python(item.name): item.pic
        for item in _walk_data_items(program.all_data_items)
        if item.pic
    }
    try:
        for para in program.paragraphs:
            method_lines.append(f'    def {_cobol_name_to_python(para.name)}(self):')
            raw_items = _split_block_items(para.raw_body) if para.raw_body.strip() else [s.raw for s in para.statements]
            if not raw_items:
                method_lines.append('        pass')
            for raw_stmt in raw_items:
                stmt = _statement_from_raw(raw_stmt)
                total_stmts += 1
                py_line, ok = translate_statement(stmt)
                if ok:
                    handled_stmts += 1
                else:
                    unhandled.append(stmt.raw)
                for line in (py_line.splitlines() if py_line else ['pass']):
                    method_lines.append(f'        {line}')
            method_lines.append('')  # blank line between methods
    finally:
        _CURRENT_CONDITION_CHECKS = {}
        _CURRENT_CONDITION_ASSIGNMENTS = {}
        _CURRENT_GROUP_FIELDS = set()
        _CURRENT_FIELD_TYPES = {}
        _CURRENT_FIELD_PICS = {}
        _TRANSLATION_CONTEXT_LOCK.release()

    # main() runner method
    if program.entry_paragraph:
        entry = _cobol_name_to_python(program.entry_paragraph.name)
        method_lines.append('    def run(self):')
        method_lines.append(f'        self.{entry}()')
        method_lines.append('')

    full_source = data_section + '\n\n' + '\n'.join(method_lines)
    full_source += f'\n\n\nif __name__ == "__main__":\n'
    full_source += f'    instance = {_make_class_name(program.program_id)}()\n'
    full_source += f'    instance.run()\n'

    coverage = (handled_stmts / total_stmts) if total_stmts > 0 else 1.0
    return _dedup_imports(full_source), coverage, unhandled


# ══════════════════════════════════════════════════════
# PHASE 3: OLLAMA LLM FALLBACK
# ══════════════════════════════════════════════════════

_LLM_SYSTEM = (
    'You are a senior software engineer specialising in COBOL-to-Python migration. '
    'You produce clean, idiomatic Python 3.10+ code. '
    'You always respond with ONLY the corrected/completed Python code. '
    'No explanations, no markdown fences, no preamble.'
)

_LLM_PROMPT = '''
The following Python code was auto-generated from COBOL by a rule-based engine.
Some statements could not be translated and are marked with # TODO [OLLAMA]:.

Your task: Replace every # TODO [OLLAMA]: comment with correct Python code.
Keep all existing translated code unchanged.
Use the same variable names (self.xxx) as the existing code.
Do NOT add any imports — they are already at the top of the file.

ORIGINAL COBOL PROGRAM:
```
{cobol_source}
```

PARTIAL PYTHON (fix the TODO lines only):
```python
{partial_python}
```

Return the complete, corrected Python file with all TODOs resolved.
'''


def llm_translate_gaps(
    partial_python: str,
    cobol_source: str,
    program_id: str,
) -> str:
    """
    Ask Ollama to fill in any # TODO [OLLAMA]: statements in the partial
    Python output from the rule-based engine.
    Returns the (hopefully) complete Python source.
    Falls back to the partial_python unchanged on any Ollama error.
    """
    prompt = _LLM_PROMPT.format(
        cobol_source=cobol_source[:4000],   # cap to save tokens
        partial_python=partial_python[:6000],
    )
    try:
        result = ollama_client.generate(
            model=config.MODEL_TRANSLATION,
            prompt=prompt,
            system=_LLM_SYSTEM,
            temperature=0.05,
        )
        # Strip markdown fences if the model added them
        result = re.sub(r'^```python\n?', '', result, flags=re.MULTILINE)
        result = re.sub(r'^```\n?', '',    result, flags=re.MULTILINE)
        logger.info('LLM gap-fill succeeded for %s (%d chars)', program_id, len(result))
        return result.strip()
    except OllamaError as exc:
        logger.warning('Ollama unavailable for %s: %s — using partial output', program_id, exc)
        return partial_python
    except Exception as exc:
        logger.warning('LLM translation error for %s: %s', program_id, exc)
        return partial_python


def _llm_style_paragraph_name(name: str) -> str:
    """
    Approximate the paragraph naming style LLM completions tend to emit.

    Unlike _cobol_name_to_python(), this deliberately does not prefix digit-led
    names, which lets us rewrite those references back to the rule engine's
    sanitized method names after gap filling.
    """
    py_name = re.sub(r'[^a-z0-9_]+', '_', name.lower().replace('-', '_'))
    return py_name.strip('_') or 'item'


def _normalise_llm_paragraph_references(source: str, program: COBOLProgram) -> str:
    """Rewrite LLM paragraph references to the deterministic sanitized names."""
    repaired = source
    for paragraph in program.paragraphs:
        llm_name = _llm_style_paragraph_name(paragraph.name)
        safe_name = _cobol_name_to_python(paragraph.name)
        if llm_name == safe_name:
            continue
        repaired = re.sub(
            rf'(\bdef\s+){re.escape(llm_name)}(\s*\()',
            rf'\1{safe_name}\2',
            repaired,
        )
        repaired = re.sub(
            rf'(\bself\.){re.escape(llm_name)}(\s*\()',
            rf'\1{safe_name}\2',
            repaired,
        )
        repaired = re.sub(
            rf'([\'"])({re.escape(llm_name)})([\'"])',
            lambda m, safe=safe_name: f'{m.group(1)}{safe}{m.group(3)}',
            repaired,
        )
    return repaired

# ══════════════════════════════════════════════════════
# PHASE 1: KB CACHE  |  PHASE 4: SAVE + WRITE  |  MAIN FUNCTION
# ══════════════════════════════════════════════════════

def translate_module(
    program_or_module,
    program_or_kb,
    kb:         KnowledgeBase | None = None,
    use_llm:    bool = True,
    output_dir: str | None = None,
) -> TranslationResult:
    """
    Translate a single COBOLProgram through all four phases.
    Returns a TranslationResult for the Validation Agent (Step 5).
    """
    if isinstance(program_or_module, COBOLProgram):
        program = program_or_module
        kb_obj = program_or_kb
    else:
        program = program_or_kb
        kb_obj = kb

    if not isinstance(program, COBOLProgram):
        raise TypeError('translate_module requires a COBOLProgram')
    if not isinstance(kb_obj, KnowledgeBase):
        raise TypeError('translate_module requires a KnowledgeBase')

    out_dir = Path(output_dir or config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{_output_basename(program)}.py'
    needs_pointer_support = bool(re.search(r'\bPOINTER\b|\bADDRESS\s+OF\b', program.raw_source, re.IGNORECASE))
    needs_coercion_support = bool(re.search(r'\bPIC\s+9', program.raw_source, re.IGNORECASE))
    needs_display_support = _needs_display_support(program)
    needs_condition_support = bool(
        re.search(
            r'\bEVALUATE\b|\bVALUES?\s+ARE\b|\bIS\s+(?:NOT\s+)?(?:ALPHABETIC|NUMERIC|POSITIVE|NEGATIVE|ZERO)\b',
            program.raw_source,
            re.IGNORECASE,
        )
    )
    needs_file_support = bool(
        re.search(r'\bFILE-CONTROL\b|\bFD\b|\bOPEN\s+INPUT\b|\bREAD\b|\bCLOSE\b', program.raw_source, re.IGNORECASE)
    )
    expected_entry = _cobol_name_to_python(program.entry_paragraph.name) if program.entry_paragraph else ''

    def _is_viable_python(source: str) -> bool:
        if '# TODO [OLLAMA]:' in source:
            return False
        try:
            compile(source, str(out_path), 'exec')
        except SyntaxError:
            return False
        if expected_entry:
            if f'def {expected_entry}(self):' not in source:
                return False
            if 'def run(self):' not in source:
                return False
            if 'if __name__ == "__main__":' not in source:
                return False
            if 'instance.run()' not in source:
                return False
        return True

    # ── Phase 1: KB Cache Check ─────────────────────────
    cobol_hash = kb_obj.hash_cobol(program.raw_source)
    cached = kb_obj.lookup_translation(program.raw_source)
    if cached:
        code = cached.translated_code or ""
        if len(code.strip()) > 50:
            logger.info(
                '[%s] CACHE HIT: reusing KB translation (confidence=1.0, method=%s, chars=%d)',
                program.program_id, cached.method or 'hybrid', len(code),
            )
            out_path.write_text(code, encoding='utf-8')
            kb_obj.record_cache_hit(program.program_id, 1.0)
            return TranslationResult(
                program_id      = program.program_id,
                source_filepath = program.filepath,
                output_filepath = str(out_path),
                translated_code = code,
                method          = 'cache',
                confidence      = CONF_CACHE,
                rule_coverage   = 1.0,
                cobol_hash      = cobol_hash,
            )
        else:
            logger.info('[%s] Cache entry too short (%d chars) — retranslating',
                        program.program_id, len(code))
            cached = None

    logger.info('%s: translating (hash %s...)', program.program_id, cobol_hash[:8])

    # ── Phase 2: Rule-Based Translation ─────────────────
    partial_python, coverage, unhandled = rule_translate_program(program)
    warnings: list[str] = []

    if unhandled:
        logger.info('%s: rule engine left %d unhandled statements — sending to LLM',
                    program.program_id, len(unhandled))
        for u in unhandled[:5]:   # log first 5
            logger.debug('  unhandled: %s', u)
        if len(unhandled) > 5:
            logger.debug('  ... and %d more', len(unhandled) - 5)

    # ── Phase 3: Ollama LLM Fallback ──────────────────
    if unhandled and use_llm:
        final_python = llm_translate_gaps(partial_python, program.raw_source,
                                          program.program_id)
        final_python = _normalise_llm_paragraph_references(final_python, program)
        method = 'hybrid' if coverage < 1.0 else 'rule'
        confidence = CONF_LLM if coverage < 0.5 else (CONF_RULE + CONF_LLM) / 2
        warnings.append(f'{len(unhandled)} statements sent to Ollama LLM')
    else:
        final_python = partial_python
        method = 'rule'
        confidence = CONF_RULE

    # Add CALL stubs warning
    if any('# TODO [STUB]' in line for line in final_python.splitlines()):
        warnings.append('CALL stubs generated — implement inter-program calls')

    # ── Phase 4: Save + Write ───────────────────────
    out_path.write_text(final_python, encoding='utf-8')
    logger.info('%s: written to %s', program.program_id, out_path)

    kb_obj.save_translation(TranslationRecord(
        cobol_hash      = cobol_hash,
        program_id      = program.program_id,
        cobol_code      = program.raw_source,
        translated_code = final_python,
        output_filepath = str(out_path),
        language        = config.TARGET_LANGUAGE,
        success         = True,
        accuracy_score  = confidence,
        method          = method,
    ))

    return TranslationResult(
        program_id      = program.program_id,
        source_filepath = program.filepath,
        output_filepath = str(out_path),
        translated_code = final_python,
        method          = method,
        confidence      = round(confidence, 4),
        rule_coverage   = round(coverage, 4),
        warnings        = warnings,
        cobol_hash      = cobol_hash,
    )

# ══════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════

def translate(
    migration_plan:  MigrationPlan,
    parsed_programs: dict[str, COBOLProgram],
    kb:              KnowledgeBase | None = None,
    use_llm:         bool = True,
    output_dir:      str | None = None,
) -> list[TranslationResult]:
    """
    Main Translation Agent entry point.
    Processes every module in the MigrationPlan in priority order.

    Parameters
    ----------
    migration_plan  : MigrationPlan from the MAPE-K planning controller
    parsed_programs : dict of program_id -> COBOLProgram from Comprehension Agent
    kb              : KnowledgeBase instance (creates default if None)
    use_llm         : if False, skip Ollama calls (useful for testing)
    output_dir      : where to write .py files (default: config.OUTPUT_DIR)

    Returns
    -------
    list[TranslationResult] consumed by Validation Agent (Step 5)
    """
    config.setup()
    kb = kb or KnowledgeBase()
    logger.info('=== Translation Agent starting: %d modules ===',
                migration_plan.total_programs)

    results: list[TranslationResult] = []

    for module in migration_plan.modules:
        pid = module.program_id
        if pid not in parsed_programs:
            logger.warning('Program %s in plan but not in parsed_programs — skipping', pid)
            continue

        program = parsed_programs[pid]
        logger.info('[%d/%d] Translating %s (composite=%.4f, method=hybrid)',
                    module.priority, migration_plan.total_programs,
                    pid, module.composite)

        result = translate_module(program, kb, use_llm=use_llm,
                                 output_dir=output_dir)
        results.append(result)

        logger.info('  -> method=%s  confidence=%.2f  coverage=%.2f',
                    result.method, result.confidence, result.rule_coverage)
        if result.warnings:
            for w in result.warnings:
                logger.warning('  WARN: %s', w)

    logger.info('=== Translation Agent done: %d results ===', len(results))
    return results


def summarise_results(results: list[TranslationResult]) -> str:
    """Human-readable summary of all translation results."""
    lines = [f'Translation Results: {len(results)} modules']
    lines.append('')
    lines.append(f'  {"Program":<14} {"Method":<8} {"Conf":>6} {"Cov":>6}  Output')
    lines.append('  ' + '-'*72)
    for r in results:
        lines.append(
            f'  {r.program_id:<14} {r.method:<8} '
            f'{r.confidence:>6.2f} {r.rule_coverage:>6.2f}  '
            f'{Path(r.output_filepath).name}'
        )
    return '\n'.join(lines)
