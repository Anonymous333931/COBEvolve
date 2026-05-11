"""
utils/cobol_parser.py -- COBOL source file parser.


Parses COBOL into:
  COBOLProgram  -- top-level container
    DataItem    -- WORKING-STORAGE / LINKAGE SECTION items
    Paragraph   -- PROCEDURE DIVISION sections
      Statement -- individual COBOL verbs (MOVE, COMPUTE, etc.)


Design: deliberately lenient. Unknown statements stored verbatim
so the LLM layer can handle them.
"""


from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def _path_to_program_id(path: str | Path) -> str:
    """Derive a stable fallback program id from the source filename."""
    stem = Path(path).stem.upper()
    stem = re.sub(r'[^A-Z0-9]+', '-', stem).strip('-')
    return stem or 'UNKNOWN'


def _uniquify_program_ids(programs: list["COBOLProgram"]) -> list["COBOLProgram"]:
    """Ensure repo-wise scans do not silently drop duplicate program IDs."""
    seen: dict[str, int] = {}
    for program in programs:
        base_id = program.program_id or 'UNKNOWN'
        if base_id == 'UNKNOWN':
            base_id = _path_to_program_id(program.filepath)
        count = seen.get(base_id, 0)
        seen[base_id] = count + 1
        if count == 0:
            program.program_id = base_id
            continue
        suffix = _path_to_program_id(program.filepath)
        candidate = f'{base_id}__{suffix}'
        extra = 1
        while candidate in seen:
            extra += 1
            candidate = f'{base_id}__{suffix}_{extra}'
        seen[candidate] = 1
        program.program_id = candidate
    return programs




# ═══════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════


@dataclass
class DataItem:
    level: int               # 01, 05, 10, 77 ...
    name: str                # COBOL name e.g. WS-ACCOUNT-NO
    pic: Optional[str]       # PIC clause e.g. X(10), 9(5)V99
    value: Optional[str]     # VALUE clause
    occurs: Optional[int | str]    # OCCURS N TIMES / OCCURS WS-SIZE TIMES
    redefines: Optional[str] # REDEFINES target name
    children: list[DataItem] = field(default_factory=list)


    @property
    def python_type(self) -> str:
        """Best-guess Python type from PIC clause."""
        if not self.pic:
            return 'dict'
        p = self.pic.upper().replace(' ', '').rstrip('.')
        if p == 'POINTER':
            return 'Any'
        if p.startswith(('X', 'A')):
            return 'str'
        if any(token in p for token in ('Z', '/', 'CR', 'DB', '$', '*')):
            return 'str'
        if 'V' in p or '.' in p:
            return 'Decimal'
        return 'int'


    @property
    def python_name(self) -> str:
        """COBOL-WS-NAME -> cobol_ws_name (snake_case)."""
        return self.name.lower().replace('-', '_')




@dataclass
class Statement:
    verb: str       # MOVE, COMPUTE, PERFORM, IF, CALL ...
    raw: str        # original statement text
    args: list[str] = field(default_factory=list)




@dataclass
class Paragraph:
    name: str
    statements: list[Statement] = field(default_factory=list)
    raw_body: str = ''
    performs: list[str] = field(default_factory=list)   # paragraphs PERFORMed
    calls: list[str] = field(default_factory=list)      # external programs CALLed




@dataclass
class COBOLProgram:
    filepath: str
    relative_path: str = ''
    program_id: str = 'UNKNOWN'
    author: str = ''
    file_section: list[DataItem] = field(default_factory=list)
    working_storage: list[DataItem] = field(default_factory=list)
    linkage_section: list[DataItem] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    raw_source: str = ''


    @property
    def paragraph_names(self) -> list[str]:
        return [p.name for p in self.paragraphs]


    @property
    def all_data_items(self) -> list[DataItem]:
        return self.file_section + self.working_storage + self.linkage_section


    @property
    def entry_paragraph(self) -> Optional[Paragraph]:
        return self.paragraphs[0] if self.paragraphs else None




# ═══════════════════════════════════════════════════════════════
# INTERNAL PARSING HELPERS
# ═══════════════════════════════════════════════════════════════


def _clean_source(src: str) -> str:
    """
    Remove COBOL sequence numbers (cols 1-6) and comment lines (col 7 = '*').
    Handles both fixed-format and free-format COBOL.
    """
    default_fixed_format = True
    for raw_line in src.splitlines():
        stripped = raw_line.lstrip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith('>>SOURCE FORMAT FREE'):
            default_fixed_format = False
            break
        if upper.startswith('>>SOURCE FORMAT FIXED'):
            default_fixed_format = True
            break
        if stripped.startswith('*>'):
            continue
        first_non_space = len(raw_line) - len(raw_line.lstrip(' '))
        if first_non_space < 7 and stripped[:1].isalpha():
            default_fixed_format = False
            break

    lines = []
    for line in src.splitlines():
        line = line.rstrip()
        if not line:
            lines.append('')
            continue
        fixed_prefix = line[:6]
        is_fixed_comment_or_continuation = (
            len(line) >= 7 and line[6] in ('-', '*', '/', 'D', 'd')
        )
        is_fixed_format = (
            default_fixed_format and
            len(line) >= 7 and
            re.fullmatch(r'[0-9 ]{0,6}', fixed_prefix) is not None and
            (line[6] == ' ' or is_fixed_comment_or_continuation)
        )
        if is_fixed_format:
            indicator = line[6]
            if indicator in ('*', '/'):
                continue   # comment
            content = line[7:72] if len(line) > 7 else ''
        else:
            content = line
        lines.append(content.rstrip())
    return '\n'.join(lines)




def _normalize(text: str) -> str:
    """Remove inline *> comments, collapse whitespace, uppercase."""
    text = re.sub(r'\*>.*', '', text)
    return ' '.join(text.split()).upper()




_PIC_RE = re.compile(
    r'PIC(?:TURE)?\s+IS\s+([^\s]+)|PIC(?:TURE)?\s+([^\s]+)',
    re.IGNORECASE,
)
_VALUE_RE = re.compile(
    r'VALUES?\s+ARE\s+(.+)|VALUE\s+IS\s+(.+)|VALUE\s+(.+)',
    re.IGNORECASE,
)
_OCCURS_RE = re.compile(
    r'OCCURS\s+(\d+|[A-Z][A-Z0-9\-]*)(?:\s+TIMES?)?',
    re.IGNORECASE,
)
_REDEFINES_RE = re.compile(r'REDEFINES\s+(\S+)', re.IGNORECASE)
_LEVEL_NAME_RE = re.compile(r'^(\d{1,2})\s+(\S+)', re.IGNORECASE)
_DATA_ITEM_START_RE = re.compile(r'^\s*\d{1,2}\s+\S+', re.IGNORECASE)




def _parse_data_item(line: str) -> Optional[DataItem]:
    """Parse a single DATA DIVISION line into a DataItem."""
    norm = _normalize(line)
    m = _LEVEL_NAME_RE.match(norm)
    if not m:
        return None
    level = int(m.group(1))
    name = m.group(2).rstrip('.')
    pic_m = _PIC_RE.search(norm)
    pic = (pic_m.group(1) or pic_m.group(2)) if pic_m else None
    if pic:
        pic = pic.rstrip('.')
    if pic is None and re.search(r'\bPOINTER\b', norm, re.IGNORECASE):
        pic = 'POINTER'
    value_m = _VALUE_RE.search(line)
    value_raw = (
        (value_m.group(1) or value_m.group(2) or value_m.group(3))
        if value_m else None
    )
    value = value_raw.strip().strip('.') if value_raw else None
    occurs_m = _OCCURS_RE.search(norm)
    if occurs_m:
        occurs_raw = occurs_m.group(1)
        occurs = int(occurs_raw) if occurs_raw.isdigit() else occurs_raw
    else:
        occurs = None
    redefines_m = _REDEFINES_RE.search(norm)
    redefines = redefines_m.group(1) if redefines_m else None
    return DataItem(level=level, name=name, pic=pic, value=value,
                   occurs=occurs, redefines=redefines)


def _iter_data_item_lines(section_text: str) -> list[str]:
    """Merge wrapped DATA DIVISION declarations into logical item lines."""
    logical_lines: list[str] = []
    current: list[str] = []
    for raw_line in section_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if _DATA_ITEM_START_RE.match(line):
            if current:
                logical_lines.append(' '.join(part.strip() for part in current))
            current = [line.strip()]
            continue
        if current:
            current.append(line.strip())
    if current:
        logical_lines.append(' '.join(part.strip() for part in current))
    return logical_lines




def _build_hierarchy(items: list[DataItem]) -> list[DataItem]:
    """Convert flat data item list to parent-child tree by level numbers."""
    root: list[DataItem] = []
    stack: list[DataItem] = []
    for item in items:
        if item.level == 78:
            root.append(item)
            continue
        while stack and stack[-1].level >= item.level:
            stack.pop()
        if stack:
            stack[-1].children.append(item)
        else:
            root.append(item)
        stack.append(item)
    return root




_VERB_RE = re.compile(
    r'^(MOVE|COMPUTE|PERFORM|IF|ELSE|END-IF|CALL|DISPLAY|ACCEPT|'
    r'ADD|SUBTRACT|MULTIPLY|DIVIDE|STOP|GOBACK|EVALUATE|WHEN|'
    r'END-EVALUATE|END-PERFORM|END-READ|END-STRING|STRING|UNSTRING|INITIALIZE|SET|OPEN|CLOSE|'
    r'READ|WRITE|REWRITE|DELETE|START|SEARCH|SORT|MERGE|INSPECT|'
    r'GO\s+TO|EXIT|CONTINUE|NEXT\s+SENTENCE)',
    re.IGNORECASE,
)




def _split_inline_statements(line: str, verb_re: re.Pattern[str], normalize) -> list[str]:
    """Split same-line COBOL verbs only when a period is a statement terminator."""
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
            if remainder and verb_re.match(normalize(remainder)):
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


def _parse_statements(body: str) -> tuple[list[Statement], list[str], list[str]]:
    """
    Parse paragraph body into (statements, performs, calls).
    statements -- all COBOL statements found
    performs   -- paragraph names called via PERFORM
    calls      -- external programs called via CALL
    """
    def _has_explicit_end(norms: list[str], start_idx: int, end_token: str) -> bool:
        return any(norm.startswith(end_token) for norm in norms[start_idx + 1:])

    def _is_perform_block_start(norm: str) -> bool:
        return bool(
            re.match(r'^PERFORM\s+UNTIL\b', norm) or
            re.match(r'^PERFORM\s+VARYING\b', norm) or
            re.match(r'^PERFORM\s+.+\s+TIMES(?:\s|$)', norm)
        )

    def _collect_nested_block(
        lines: list[str],
        start_idx: int,
        start_predicate,
        end_token: str,
    ) -> tuple[str, int]:
        depth = 1
        chunk = [lines[start_idx]]
        i = start_idx + 1
        while i < len(lines):
            current = lines[i]
            current_norm = _normalize(current)
            if start_predicate(current_norm):
                depth += 1
            elif current_norm.startswith(end_token):
                depth -= 1
            chunk.append(current)
            i += 1
            if depth == 0:
                break
        return '\n'.join(chunk).strip(), i

    def _split_statement_chunks(text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                continue
            lines.extend(_split_inline_statements(line, _VERB_RE, _normalize))
        norms = [_normalize(line) for line in lines]

        def _consume_simple_statement(start_idx: int) -> tuple[str, int]:
            chunk = [lines[start_idx]]
            i = start_idx + 1
            while i < len(lines):
                if _VERB_RE.match(norms[i]):
                    break
                chunk.append(lines[i])
                i += 1
            return '\n'.join(chunk).strip(), i

        def _consume_statement(start_idx: int) -> tuple[str, int]:
            norm = norms[start_idx]
            if norm.startswith('IF '):
                return _consume_if(start_idx)
            if norm.startswith('EVALUATE ') and _has_explicit_end(norms, start_idx, 'END-EVALUATE'):
                return _collect_nested_block(
                    lines, start_idx, lambda candidate: candidate.startswith('EVALUATE '), 'END-EVALUATE'
                )
            if norm.startswith('READ ') and _has_explicit_end(norms, start_idx, 'END-READ'):
                return _collect_nested_block(
                    lines, start_idx, lambda candidate: candidate.startswith('READ '), 'END-READ'
                )
            if norm.startswith('STRING ') and _has_explicit_end(norms, start_idx, 'END-STRING'):
                return _collect_nested_block(
                    lines, start_idx, lambda candidate: candidate.startswith('STRING '), 'END-STRING'
                )
            if _is_perform_block_start(norm) and _has_explicit_end(norms, start_idx, 'END-PERFORM'):
                return _collect_nested_block(
                    lines, start_idx, _is_perform_block_start, 'END-PERFORM'
                )
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
                        return '\n'.join(chunk).strip(), i
                    next_norm = norms[i]
                    if not (next_norm.startswith('ELSE') or next_norm.startswith('END-IF')):
                        return '\n'.join(chunk).strip(), i

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
                            return '\n'.join(chunk).strip(), i
                        if not norms[i].startswith('END-IF'):
                            return '\n'.join(chunk).strip(), i

            if i < len(lines) and norms[i].startswith('END-IF'):
                chunk.append(lines[i])
                i += 1
            return '\n'.join(chunk).strip(), i

        items: list[str] = []
        i = 0
        while i < len(lines):
            chunk, i = _consume_statement(i)
            items.append(chunk)
        return items

    statements, performs, calls = [], [], []
    for sent in _split_statement_chunks(body):
        if not sent:
            continue
        norm = _normalize(sent)
        if not norm:
            continue
        m = _VERB_RE.match(norm)
        verb = m.group(0).strip() if m else 'UNKNOWN'
        args_str = norm[len(verb):].strip() if m else norm
        stmt = Statement(
            verb=verb,
            raw=sent,
            args=[part.rstrip('.') for part in args_str.split()] if args_str else [],
        )
        statements.append(stmt)
        if verb == 'PERFORM':
            t = re.match(r'PERFORM\s+(\S+)', norm, re.IGNORECASE)
            if t and t.group(1).upper() not in ('VARYING', 'UNTIL', 'THROUGH', 'THRU', 'WITH', 'TEST'):
                performs.append(t.group(1).rstrip('.'))
        elif verb == 'CALL':
            c = re.match(r'CALL\s+[\'"]?([A-Z0-9\-]+)[\'"]?(?:\s|$|\.)', norm, re.IGNORECASE)
            if c:
                calls.append(c.group(1))
    return statements, performs, calls




# ═══════════════════════════════════════════════════════════════
# DIVISION SPLITTER
# ═══════════════════════════════════════════════════════════════


_DIV_RE = re.compile(
    r'(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION',
    re.IGNORECASE,
)

_PARAGRAPH_HEADER_RE = re.compile(
    r'^\s*([A-Z0-9][A-Z0-9\-]*)\s*(?:SECTION)?\.\s*$',
    re.IGNORECASE,
)

_RESERVED_PARAGRAPH_NAMES = {
    'MOVE', 'COMPUTE', 'PERFORM', 'IF', 'ELSE', 'END-IF', 'CALL', 'DISPLAY',
    'ACCEPT', 'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'STOP', 'GOBACK',
    'EVALUATE', 'WHEN', 'END-EVALUATE', 'STRING', 'UNSTRING', 'INITIALIZE',
    'SET', 'OPEN', 'CLOSE', 'READ', 'WRITE', 'REWRITE', 'DELETE', 'START',
    'SEARCH', 'SORT', 'MERGE', 'INSPECT', 'GO', 'EXIT', 'CONTINUE', 'NEXT',
    'END-READ', 'END-STRING', 'END-PERFORM', 'AT', 'NOT',
}




def parse_cobol_source(source: str, filepath: str = '<memory>') -> COBOLProgram:
    """Parse COBOL source from a string into a COBOLProgram object."""
    program = COBOLProgram(filepath=filepath, raw_source=source)
    cleaned = _clean_source(source)


    # -- Find division boundaries --
    div_pos = [(m.start(), m.group(1).upper()) for m in _DIV_RE.finditer(cleaned)]


    def _get_div(name: str) -> str:
        for i, (pos, div) in enumerate(div_pos):
            if div == name:
                end = div_pos[i+1][0] if i+1 < len(div_pos) else len(cleaned)
                return cleaned[pos:end]
        return ''


    id_div = _get_div('IDENTIFICATION')
    data_div = _get_div('DATA')
    proc_div = _get_div('PROCEDURE')

    if not data_div:
        proc_start = cleaned.upper().find('PROCEDURE DIVISION')
        pre_procedure = cleaned[:proc_start] if proc_start != -1 else cleaned
        if re.search(
            r'\b(?:FILE|WORKING-STORAGE|LINKAGE)\s+SECTION\b',
            pre_procedure,
            re.IGNORECASE,
        ):
            data_div = pre_procedure


    # -- IDENTIFICATION DIVISION --
    pid = re.search(r'PROGRAM-ID\.?\s+([A-Z0-9\-]+)', id_div, re.IGNORECASE)
    if not pid:
        pid = re.search(r'PROGRAM-ID\.?\s+([A-Z0-9\-]+)', cleaned, re.IGNORECASE)
    if pid:
        program.program_id = pid.group(1).rstrip('.')
    auth = re.search(r'AUTHOR\.?\s+(.+?)(?=\n[A-Z]|\Z)', id_div, re.IGNORECASE | re.DOTALL)
    if auth:
        program.author = auth.group(1).strip()


    # -- DATA DIVISION --
    if data_div:
        fs = re.search(r'FILE\s+SECTION(.+?)(?=\w+\s+(?:SECTION|DIVISION)|\Z)',
                       data_div, re.IGNORECASE | re.DOTALL)
        ws = re.search(r'WORKING-STORAGE\s+SECTION(.+?)(?=\w+\s+(?:SECTION|DIVISION)|\Z)',
                       data_div, re.IGNORECASE | re.DOTALL)
        lk = re.search(r'LINKAGE\s+SECTION(.+?)(?=\w+\s+(?:SECTION|DIVISION)|\Z)',
                       data_div, re.IGNORECASE | re.DOTALL)
        for sect, target in [
            (fs, program.file_section),
            (ws, program.working_storage),
            (lk, program.linkage_section),
        ]:
            if sect:
                flat = [_parse_data_item(ln) for ln in _iter_data_item_lines(sect.group(1))]
                flat = [x for x in flat if x is not None]
                target.extend(_build_hierarchy(flat))


    # -- PROCEDURE DIVISION --
    if proc_div:
        body = re.sub(r'PROCEDURE\s+DIVISION[^.]*\.', '', proc_div,
                      flags=re.IGNORECASE, count=1)
        body_lines = body.splitlines()
        header_candidates: list[tuple[int, str, int]] = []
        for idx, line in enumerate(body_lines):
            match = _PARAGRAPH_HEADER_RE.match(line)
            if not match:
                continue
            name = match.group(1).strip().upper()
            if name in _RESERVED_PARAGRAPH_NAMES:
                continue
            indent = len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())
            header_candidates.append((idx, name, indent))
        header_indent = min((indent for _, _, indent in header_candidates), default=None)
        headers = [
            (idx, name)
            for idx, name, indent in header_candidates
            if header_indent is None or indent == header_indent
        ]

        def _append_paragraph(name: str, lines: list[str]):
            raw_body = '\n'.join(lines).strip()
            if not raw_body:
                return
            stmts, perfs, cals = _parse_statements(raw_body)
            program.paragraphs.append(Paragraph(
                name=name,
                statements=stmts,
                raw_body=raw_body,
                performs=perfs,
                calls=cals,
            ))

        if not headers:
            _append_paragraph('MAIN-PROCEDURE', body_lines)
        else:
            preamble = body_lines[:headers[0][0]]
            if any(line.strip() for line in preamble):
                entry_name = 'MAIN-PROCEDURE'
                existing = {name for _, name in headers}
                if entry_name in existing:
                    entry_name = 'ENTRY-PROCEDURE'
                _append_paragraph(entry_name, preamble)
            for i, (start_idx, name) in enumerate(headers):
                end_idx = headers[i + 1][0] if i + 1 < len(headers) else len(body_lines)
                _append_paragraph(name, body_lines[start_idx + 1:end_idx])


    logger.debug('Parsed %s: %d data items, %d paragraphs',
                 program.program_id, len(program.all_data_items), len(program.paragraphs))
    return program




def parse_cobol_file(filepath: str | Path) -> COBOLProgram:
    """Parse a COBOL file from disk."""
    fp = Path(filepath)
    src = fp.read_text(encoding='utf-8', errors='replace')
    program = parse_cobol_source(src, str(fp))
    program.relative_path = fp.name
    return program




def parse_cobol_directory(
    directory: str | Path,
    repo_root: str | Path | None = None,
) -> list[COBOLProgram]:
    """Parse all COBOL files in a directory (recursive)."""
    programs = []
    root = Path(repo_root) if repo_root is not None else None
    for ext in ('*.cbl', '*.cob', '*.cobol', '*.CBL', '*.COB'):
        for fp in Path(directory).rglob(ext):
            try:
                program = parse_cobol_file(fp)
                if root is not None:
                    try:
                        program.relative_path = str(fp.relative_to(root))
                    except ValueError:
                        program.relative_path = fp.name
                programs.append(program)
            except Exception as exc:
                logger.warning('Failed to parse %s: %s', fp, exc)
    return _uniquify_program_ids(programs)


def parse_multiple_repos(
    repo_paths: list[str | Path],
) -> dict[str, list[COBOLProgram]]:
    """Parse multiple repo roots and return programs keyed by repo name."""
    results: dict[str, list[COBOLProgram]] = {}
    for repo_path in repo_paths:
        repo = Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(f'Repo not found: {repo}')
        if not repo.is_dir():
            raise NotADirectoryError(f'Repo is not a directory: {repo}')
        results[repo.name] = parse_cobol_directory(repo, repo_root=repo)
    return results
