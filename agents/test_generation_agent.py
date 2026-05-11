
# ── FILE: cobol_moderniser/agents/test_generation_agent.py ────────────────
"""
agents/test_generation_agent.py — Test Generation Agent

Generates behavioural test cases by:
  1. Compiling and running original COBOL via GnuCOBOL
  2. Augmenting with LLM-generated edge-case inputs

Grounded in: COBug (ICSE 2026 under review)
"""
from __future__ import annotations
import hashlib, json, logging, os, re, shlex, shutil, subprocess, tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

from config import config
from utils.cobol_parser import COBOLProgram, parse_cobol_directory
from utils.ollama_client import ollama_client, OllamaError

logger = logging.getLogger(__name__)

_NONSTANDALONE_SOURCE_REASON = 'Not a standalone executable program'
_INTERACTIVE_ORACLE_REASON = 'Not suitable for non-interactive oracle execution'

# ── Data models ───────────────────────────────────────────
def _oracle_basename(program: COBOLProgram) -> str:
    """Prefer the source filename when it disambiguates duplicate PROGRAM-IDs."""
    program_slug = re.sub(r'[^a-z0-9]+', '-', program.program_id.lower()).strip('-')
    source_slug = re.sub(r'[^a-z0-9]+', '-', Path(program.filepath).stem.lower()).strip('-')
    return source_slug if source_slug and source_slug != program_slug else (program_slug or source_slug or 'module')


@dataclass
class TestCase:
    test_id: str               # 'tc_001', 'tc_002', ...
    inputs: dict               # {var_name: value} fed via stdin to both COBOL and Python
    expected_output: str       # stdout from GnuCOBOL run
    description: str = ''
    source: str = 'oracle'     # 'oracle' | 'llm_augment'
    __test__ = False

@dataclass
class BehavioralOracle:
    program_id: str
    cobol_filepath: str
    test_cases: list = field(default_factory=list)
    compile_success: bool = False
    compile_error: str = ''
    total_cases: int = 0
    generated_at: str = ''
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()
        self.total_cases = len(self.test_cases)
    def to_dict(self) -> dict:
        return {
            'program_id': self.program_id,
            'cobol_filepath': self.cobol_filepath,
            'test_cases': [
                {'test_id': tc.test_id, 'inputs': tc.inputs,
                 'expected_output': tc.expected_output,
                 'description': tc.description, 'source': tc.source}
                for tc in self.test_cases],
            'compile_success': self.compile_success,
            'compile_error': self.compile_error,
            'total_cases': self.total_cases,
            'generated_at': self.generated_at,
        }

# ════════════════════════════════════════════════
# PHASE 1 + 3: COMPILE AND RUN COBOL
# ════════════════════════════════════════════════

def _gnucobol_available() -> bool:
    """Check whether cobc is installed and accessible."""
    return shutil.which(config.GNUCOBOL_PATH or 'cobc') is not None

def _read_source_text(cobol_filepath: str) -> str:
    try:
        return Path(cobol_filepath).read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''


def _strip_comment_lines(source_text: str) -> str:
    """Drop comment/documentation lines before lightweight preflight checks."""
    cleaned_lines: list[str] = []
    for line in source_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('*>') or stripped.startswith('*'):
            continue
        if len(line) > 6 and line[6] in ('*', '/'):
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def _normalise_compile_source_text(source_text: str) -> str:
    """
    Apply small, local source normalisations for oracle compilation only.

    These fixes are intentionally narrow and only affect the temporary source
    copy used for GnuCOBOL runs; the original repo files remain untouched.
    """
    normalised = source_text
    if normalised and not normalised.endswith('\n'):
        normalised += '\n'

    lines = normalised.splitlines(keepends=True)
    rewritten: list[str] = []
    inside_config_section = False
    saw_special_names = False

    def _is_comment_or_blank(line_text: str) -> bool:
        stripped = line_text.strip()
        return not stripped or stripped.startswith('*>') or stripped.startswith('*')

    def _is_boundary_line(line_text: str) -> bool:
        stripped = line_text.strip().upper()
        return bool(
            re.match(r'^[A-Z0-9-]+\s+DIVISION\.$', stripped) or
            re.match(r'^[A-Z0-9-]+\s+SECTION\.$', stripped) or
            stripped.startswith('END PROGRAM')
        )

    def _needs_terminating_period(line_text: str) -> bool:
        stripped = line_text.strip().upper()
        if not stripped or stripped.endswith('.'):
            return False
        return bool(re.match(r'^(\d{2}|FD|SD|SELECT|DECIMAL-POINT|SPECIAL-NAMES)\b', stripped))

    def _terminate_previous_clause() -> None:
        for idx in range(len(rewritten) - 1, -1, -1):
            prior = rewritten[idx]
            if _is_comment_or_blank(prior):
                continue
            if _needs_terminating_period(prior):
                newline = '\n' if prior.endswith('\n') else ''
                body = prior[:-1] if newline else prior
                rewritten[idx] = f'{body}.{newline}'
            break

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        is_comment = stripped.startswith('*>') or stripped.startswith('*')

        if not is_comment and _is_boundary_line(line):
            _terminate_previous_clause()

        if not is_comment and upper == 'CONFIGURATION SECTION.':
            inside_config_section = True
            saw_special_names = False
            rewritten.append(line)
            continue

        if inside_config_section and not is_comment:
            if upper == 'SPECIAL-NAMES.':
                saw_special_names = True
            elif re.match(r'^[A-Z0-9-]+\s+SECTION\.$', upper) or re.match(
                r'^[A-Z0-9-]+\s+DIVISION\.$',
                upper,
            ):
                inside_config_section = False
            elif (
                not saw_special_names and
                re.match(r'^DECIMAL-POINT\s+IS\s+COMMA\.?$', upper)
            ):
                indent = re.match(r'^(\s*)', line).group(1)
                rewritten.append(f'{indent}SPECIAL-NAMES.\n')
                saw_special_names = True

        rewritten.append(line)

    return ''.join(rewritten)


def _prepare_compile_source(
    cobol_filepath: str,
    source_text: str,
    work_dir: str,
) -> Path:
    """
    Materialise a normalised compile-only copy of the COBOL source in work_dir.
    """
    src = Path(cobol_filepath).resolve()
    prepared_source = _normalise_compile_source_text(source_text)
    prepared_path = Path(work_dir) / src.name
    prepared_path.write_text(prepared_source, encoding='utf-8')
    return prepared_path


def _oracle_preflight_issue(program: COBOLProgram, source_text: str) -> str | None:
    """
    Detect source files that are clearly not runnable standalone COBOL programs.

    These files show up often in repo-wise datasets as copybooks, section
    fragments, or subprogram entrypoints. They should be skipped fairly instead
    of being counted as ordinary compile failures.
    """
    text = _strip_comment_lines(source_text).upper()
    has_program_id = 'PROGRAM-ID.' in text
    first_procedure_division = re.search(
        r'\bPROCEDURE\s+DIVISION\b([^.]*)',
        text,
        re.IGNORECASE,
    )
    has_procedure_division = first_procedure_division is not None

    if not has_program_id and not has_procedure_division:
        return (
            f'{_NONSTANDALONE_SOURCE_REASON}: '
            'source fragment/copybook missing PROGRAM-ID and PROCEDURE DIVISION'
        )
    if not has_program_id:
        return f'{_NONSTANDALONE_SOURCE_REASON}: missing PROGRAM-ID'
    if not has_procedure_division:
        return f'{_NONSTANDALONE_SOURCE_REASON}: missing PROCEDURE DIVISION'
    if (
        first_procedure_division and
        'USING' in first_procedure_division.group(1).upper()
    ):
        return f'{_NONSTANDALONE_SOURCE_REASON}: PROCEDURE DIVISION has USING clause'
    # SCREEN SECTION/menu applications require terminal-style interaction that
    # the stdin/stdout oracle cannot drive reliably in repo-wide batch runs.
    has_screen_section = 'SCREEN SECTION.' in text
    has_screen_controls = bool(re.search(
        r'\b(BLANK\s+SCREEN|ERASE\s+EOL|BACKGROUND-COLOR|FOREGROUND-COLOR)\b',
        text,
        re.IGNORECASE,
    ))
    has_menu_screen_io = bool(re.search(
        r'\b(DISPLAY\s+[A-Z0-9-]*TELA[A-Z0-9-]*|ACCEPT\s+[A-Z0-9-]*MENU[A-Z0-9-]*)\b',
        text,
        re.IGNORECASE,
    ))
    if has_screen_section and (has_screen_controls or has_menu_screen_io):
        return (
            f'{_INTERACTIVE_ORACLE_REASON}: '
            'interactive SCREEN SECTION/menu program'
        )
    return None

def _detect_external_dependencies(source_text: str) -> list[str]:
    text = source_text.lower()
    deps: list[str] = []
    if 'raylib' in text or 'initwindow' in text or 'begindrawing' in text:
        deps.append('raylib')
    if 'sqlite' in text or '-lsqlite3' in text:
        deps.append('sqlite3')
    if 'ocsqlite_' in text or 'call "ocsqlite' in text:
        deps.append('ocsqlite')
    return sorted(dict.fromkeys(deps))

def _source_format_hint(source_text: str) -> str | None:
    if re.search(r'>>\s*SOURCE\s+FORMAT\s+IS\s+FIXED', source_text, re.IGNORECASE):
        return 'fixed'
    if re.search(r'>>\s*SOURCE\s+FORMAT\s+IS\s+FREE', source_text, re.IGNORECASE):
        return 'free'
    non_empty = [line for line in source_text.splitlines() if line.strip()]
    if non_empty:
        fixed_like = sum(1 for line in non_empty if line.startswith('      '))
        if fixed_like / len(non_empty) >= 0.6:
            return 'fixed'
    return None

def _cobc_commands(source_text: str) -> list[list[str]]:
    """Extract tokenized `cobc ...` commands embedded in source comments."""
    commands: list[list[str]] = []
    for match in re.finditer(r'cobc\s+([^\n\r]+)', source_text, re.IGNORECASE):
        try:
            commands.append(shlex.split(match.group(1)))
        except ValueError:
            continue
    return commands


def _extract_helper_compile_steps(
    cobol_filepath: str,
    source_text: str,
) -> list[tuple[str, str, list[str]]]:
    """
    Extract compile-only helper build steps, for example:
      cobc -c helper.c
      cobc -c -DFLAG helper.c

    Returns tuples of:
      (helper_source_path, output_object_name, extra_flags)
    """
    src = Path(cobol_filepath).resolve()
    steps: list[tuple[str, str, list[str]]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    for tokens in _cobc_commands(source_text):
        if '-c' not in tokens:
            continue

        helper_source: str | None = None
        output_name: str | None = None
        flags: list[str] = []
        skip_next = False
        capture_output_name = False

        for token in tokens:
            if skip_next:
                if capture_output_name:
                    output_name = Path(token).name
                skip_next = False
                capture_output_name = False
                continue
            if token == '-o':
                skip_next = True
                capture_output_name = True
                continue
            if token == '-c' or token == '-x' or token.startswith('-x'):
                continue
            if re.search(r'\.(cbl|cob|cobol)$', token, re.IGNORECASE):
                continue
            if re.search(r'\.(c|cc|cpp)$', token, re.IGNORECASE):
                helper_path = (src.parent / token).resolve()
                helper_source = str(helper_path)
                if output_name is None:
                    output_name = f'{Path(token).stem}.o'
                continue
            if token.endswith('.o'):
                output_name = Path(token).name
                continue
            flags.append(token)

        if not helper_source:
            continue
        final_output = output_name or f'{Path(helper_source).stem}.o'
        key = (helper_source, final_output, tuple(flags))
        if key in seen:
            continue
        seen.add(key)
        steps.append((helper_source, final_output, flags))

    return steps


def _extract_compile_hints(
    cobol_filepath: str,
    source_text: str,
    helper_objects: dict[str, str] | None = None,
) -> tuple[list[list[str]], list[str]]:
    """
    Extract executable-oriented cobc command hints from source comments.

    Compile-only helper steps are handled separately. The returned missing
    helper list contains helper sources/objects referenced by executable hints
    that are not available locally.
    """
    src = Path(cobol_filepath).resolve()
    variants: list[list[str]] = []
    missing_helpers: list[str] = []
    for tokens in _cobc_commands(source_text):
        if '-c' in tokens:
            continue
        args: list[str] = []
        skip_next = False
        for token in tokens:
            if skip_next:
                skip_next = False
                continue
            if token == '-o':
                skip_next = True
                continue
            if token.lower() == src.name.lower():
                continue
            if re.search(r'\.(cbl|cob|cobol)$', token, re.IGNORECASE):
                continue
            if re.search(r'\.(c|cc|cpp)$', token, re.IGNORECASE):
                helper_path = (src.parent / token).resolve()
                if helper_path.exists():
                    args.append(str(helper_path))
                else:
                    object_name = f'{Path(token).stem}.o'
                    if helper_objects and object_name in helper_objects:
                        args.append(helper_objects[object_name])
                    else:
                        missing_helpers.append(Path(token).name)
                continue
            if token.endswith('.o'):
                helper = (src.parent / token).resolve()
                if helper.exists():
                    args.append(str(helper))
                else:
                    helper_name = Path(token).name
                    if helper_objects and helper_name in helper_objects:
                        args.append(helper_objects[helper_name])
                    else:
                        missing_helpers.append(helper_name)
                continue
            args.append(token)
        if args:
            variants.append(args)
    return variants, sorted(dict.fromkeys(missing_helpers))


def _missing_helper_artifacts(cobol_filepath: str, source_text: str) -> list[str]:
    """Return helper sources/objects referenced by build hints but absent locally."""
    missing: list[str] = []

    for helper_source, _output_name, _flags in _extract_helper_compile_steps(
        cobol_filepath, source_text
    ):
        helper_path = Path(helper_source)
        if not helper_path.exists():
            missing.append(helper_path.name)

    _variants, hinted_missing = _extract_compile_hints(
        cobol_filepath, source_text, helper_objects={}
    )
    missing.extend(hinted_missing)
    return sorted(dict.fromkeys(missing))


def _infer_repo_root(program: COBOLProgram) -> Path:
    """Best-effort repo root inference from the parsed relative path."""
    src = Path(program.filepath).resolve()
    if program.relative_path:
        rel = Path(program.relative_path)
        candidate = src
        for _ in rel.parts:
            candidate = candidate.parent
        if candidate.exists():
            return candidate
    return src.parent


@lru_cache(maxsize=256)
def _repo_include_config(repo_root: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """
    Discover COPY/include search paths and useful COPY extensions inside a repo.

    We keep this cached because repo-wide scans can be expensive and the same
    repo root is reused across many modules in a single batch run.
    """
    root = Path(repo_root)
    if not root.exists():
        return (), (), ()

    include_dirs: list[str] = []
    seen_dirs: set[str] = set()
    extensions: set[str] = set()
    include_files: list[str] = []
    patterns = ('*.cpy', '*.CPY', '*.copy', '*.COPY', '*.cbl', '*.CBL', '*.cob', '*.COB')

    def add_dir(path: Path) -> None:
        resolved = str(path.resolve())
        if resolved in seen_dirs:
            return
        seen_dirs.add(resolved)
        include_dirs.append(resolved)

    add_dir(root)
    for pattern in patterns:
        for member in root.rglob(pattern):
            add_dir(member.parent)
            include_files.append(str(member.resolve()))
            suffix = member.suffix.lstrip('.')
            if suffix:
                extensions.add(suffix)

    ordered_exts = []
    for ext in ('cpy', 'CPY', 'copy', 'COPY', 'cbl', 'CBL', 'cob', 'COB', 'cobol', 'COBOL'):
        if ext in extensions or ext.lower() in {e.lower() for e in extensions}:
            ordered_exts.append(ext)
    return tuple(include_dirs), tuple(dict.fromkeys(ordered_exts)), tuple(dict.fromkeys(include_files))

def _prepare_include_alias_dir(program: COBOLProgram, work_dir: str) -> str | None:
    """
    Materialize case-insensitive include aliases in a temp dir.

    Some repos reference COPY members as upper-case `.CPY` while the files
    on disk are lower-case `.cpy`, which fails on Linux. We expose a small
    alias directory ahead of the real include dirs so GnuCOBOL can resolve
    those members without modifying the repo itself.
    """
    repo_root = _infer_repo_root(program)
    _, _, include_files = _repo_include_config(str(repo_root))
    if not include_files:
        return None

    alias_dir = Path(work_dir) / '_include_aliases'
    alias_dir.mkdir(parents=True, exist_ok=True)
    for file_name in include_files:
        member = Path(file_name)
        aliases = {
            member.name,
            member.name.upper(),
            member.name.lower(),
        }
        for alias in aliases:
            target = alias_dir / alias
            if target.exists():
                continue
            try:
                os.symlink(member, target)
            except FileExistsError:
                continue
            except OSError:
                shutil.copyfile(member, target)
    return str(alias_dir)


def _compile_support_args(program: COBOLProgram, work_dir: str | None = None) -> list[str]:
    """Build shared cobc arguments for repo-local COPY/include resolution."""
    src = Path(program.filepath).resolve()
    repo_root = _infer_repo_root(program)
    repo_dirs, copy_exts, _ = _repo_include_config(str(repo_root))
    alias_dir = _prepare_include_alias_dir(program, work_dir) if work_dir else None

    ordered_dirs: list[str] = []
    seen: set[str] = set()
    search_paths: list[Path] = []
    if alias_dir:
        search_paths.append(Path(alias_dir))
    search_paths.extend([src.parent, repo_root, *map(Path, repo_dirs)])
    for path in search_paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered_dirs.append(resolved)

    args: list[str] = []
    for directory in ordered_dirs:
        args.extend(['-I', directory])
    for ext in copy_exts:
        args.extend(['-ext', ext])
    return args


@lru_cache(maxsize=128)
def _repo_program_index(repo_root: str) -> dict[str, tuple[COBOLProgram, ...]]:
    """Parse a repo once so sibling CALL targets can be resolved locally."""
    root = Path(repo_root)
    if not root.exists() or not root.is_dir():
        return {}

    index: dict[str, list[COBOLProgram]] = {}
    for repo_program in parse_cobol_directory(root, repo_root=root):
        index.setdefault(repo_program.program_id.upper(), []).append(repo_program)
    return {key: tuple(value) for key, value in index.items()}


def _direct_calls(program: COBOLProgram) -> list[str]:
    calls: list[str] = []
    for paragraph in program.paragraphs:
        for call in paragraph.calls:
            call_name = call.upper()
            if call_name not in calls:
                calls.append(call_name)
    return calls


def _module_compile_variants(cobol_filepath: str) -> tuple[list[list[str]], list[str]]:
    variants, deps, _missing_helpers = _compile_variants(cobol_filepath)
    module_variants: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for variant in variants:
        module_variant = ['-m', *[flag for flag in variant if flag != '-x']]
        key = tuple(module_variant)
        if key in seen:
            continue
        seen.add(key)
        module_variants.append(module_variant)
    return module_variants, deps


def compile_cobol_module(program: COBOLProgram, work_dir: str) -> tuple[str | None, str]:
    """
    Compile a called COBOL subprogram as a loadable module.
    Returns (module_path, error_message).
    """
    src = Path(program.filepath).resolve()
    prepared_src = _prepare_compile_source(program.filepath, _read_source_text(program.filepath), work_dir)
    out = Path(work_dir) / program.program_id
    cobc = config.GNUCOBOL_PATH or 'cobc'
    compile_variants, deps = _module_compile_variants(program.filepath)
    support_args = _compile_support_args(program, work_dir)
    errors: list[str] = []
    for flags in compile_variants:
        try:
            result = subprocess.run(
                [cobc, *support_args, *flags, str(prepared_src), '-o', str(out)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(src.parent),
            )
            if result.returncode == 0:
                logger.info('Compiled module %s -> %s using flags %s',
                            src.name, out.name, ' '.join(flags))
                return str(out), ''
            err = (result.stderr or result.stdout or 'compile error').strip()
            errors.append(f'{" ".join(flags)}: {err[:400]}')
        except subprocess.TimeoutExpired:
            errors.append(f'{" ".join(flags)}: Compilation timed out after 30s')
        except Exception as exc:
            errors.append(f'{" ".join(flags)}: {exc}')
    logger.warning('GnuCOBOL module compile failed for %s: %s', src.name, errors[0][:200])
    return None, _summarise_compile_failures(errors, deps)


def _prepare_called_modules(program: COBOLProgram, work_dir: str) -> list[str]:
    """
    Compile repo-local CALL targets into the temp work dir so the main
    executable can load them via COB_LIBRARY_PATH at runtime.
    """
    repo_root = _infer_repo_root(program)
    program_index = _repo_program_index(str(repo_root))
    compiled: list[str] = []
    visited: set[str] = set()
    source_path = Path(program.filepath).resolve()

    def visit(call_name: str) -> None:
        key = call_name.upper()
        if key in visited:
            return
        visited.add(key)
        for candidate in program_index.get(key, ()):
            if Path(candidate.filepath).resolve() == source_path:
                continue
            module_path, _ = compile_cobol_module(candidate, work_dir)
            if module_path:
                compiled.append(module_path)
                for nested_call in _direct_calls(candidate):
                    visit(nested_call)

    for call_name in _direct_calls(program):
        visit(call_name)
    return compiled

def _compile_helper_objects(
    cobol_filepath: str,
    source_text: str,
    work_dir: str,
) -> tuple[dict[str, str], list[str], list[str]]:
    """Compile repo-local helper C sources referenced by cobc build hints."""
    cobc = config.GNUCOBOL_PATH or 'cobc'
    helper_objects: dict[str, str] = {}
    errors: list[str] = []
    missing: list[str] = []

    for helper_source, output_name, flags in _extract_helper_compile_steps(
        cobol_filepath, source_text
    ):
        helper_path = Path(helper_source)
        if not helper_path.exists():
            missing.append(helper_path.name)
            continue

        out = Path(work_dir) / output_name
        try:
            result = subprocess.run(
                [cobc, '-c', *flags, str(helper_path), '-o', str(out)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(helper_path.parent),
            )
            if result.returncode == 0:
                helper_objects[out.name] = str(out.resolve())
                logger.info('Compiled helper %s -> %s', helper_path.name, out.name)
                continue
            err = (result.stderr or result.stdout or 'compile error').strip()
            errors.append(f'helper {helper_path.name}: {err[:400]}')
        except subprocess.TimeoutExpired:
            errors.append(f'helper {helper_path.name}: compilation timed out after 30s')
        except Exception as exc:
            errors.append(f'helper {helper_path.name}: {exc}')

    return helper_objects, errors, sorted(dict.fromkeys(missing))


def _compile_variants(
    cobol_filepath: str,
    helper_objects: dict[str, str] | None = None,
) -> tuple[list[list[str]], list[str], list[str]]:
    source_text = _read_source_text(cobol_filepath)
    deps = _detect_external_dependencies(source_text)
    format_hint = _source_format_hint(source_text)
    hinted_variants, missing_helpers = _extract_compile_hints(
        cobol_filepath, source_text, helper_objects
    )

    if format_hint == 'fixed':
        base = [['-x', '-fixed'], ['-x'], ['-x', '-free']]
    elif format_hint == 'free':
        base = [['-x', '-free'], ['-x'], ['-x', '-fixed']]
    else:
        base = [['-x', '-free'], ['-x', '-fixed'], ['-x']]

    variants: list[list[str]] = []
    for hint in hinted_variants:
        hint_args = list(hint)
        if not any(token == '-x' or token.startswith('-x') for token in hint_args):
            hint_args.insert(0, '-x')
        variants.append(hint_args)
        if '-fixed' not in hint_args and format_hint == 'fixed':
            variants.append(hint_args + ['-fixed'])
        if '-free' not in hint_args and format_hint == 'free':
            variants.append(hint_args + ['-free'])

    variants.extend(base)

    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for variant in variants:
        key = tuple(variant)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped, deps, missing_helpers

def _append_dependency_hint(message: str, deps: list[str]) -> str:
    if not deps:
        return message
    hint = f'Requires external dependency(s): {", ".join(deps)}'
    return f'{message}\n{hint}' if message else hint

def _missing_link_libraries(errors: list[str]) -> list[str]:
    libs: list[str] = []
    for error in errors:
        libs.extend(re.findall(r'cannot find -l([A-Za-z0-9_\-+.]+)', error))
    return sorted(dict.fromkeys(libs))


def _missing_runtime_modules(errors: list[str]) -> list[str]:
    modules: list[str] = []
    for error in errors:
        modules.extend(re.findall(r"module '([^']+)' not found", error, re.IGNORECASE))
    return sorted(dict.fromkeys(modules))


def _primary_failure_message(errors: list[str]) -> str:
    if not errors:
        return ''
    priorities = (
        'cannot find -l',
        'undefined reference',
        "module '",
        'error:',
    )
    lowered = [error.lower() for error in errors]
    for marker in priorities:
        for idx, error in enumerate(lowered):
            if marker in error:
                return errors[idx].strip()[:500]
    return errors[0].strip()[:500]


def _summarise_compile_failures(
    errors: list[str],
    deps: list[str],
    missing_helpers: list[str] | None = None,
) -> str:
    parts: list[str] = []
    helper_list = sorted(dict.fromkeys(missing_helpers or []))
    if helper_list:
        parts.append(
            f'Missing repo-local helper source/object(s): {", ".join(helper_list)}'
        )
    missing_libs = _missing_link_libraries(errors)
    if missing_libs:
        parts.append(
            f'Missing external library/libraries: {", ".join(missing_libs)}'
        )
    if deps:
        parts.append(f'Requires external dependency(s): {", ".join(deps)}')
    primary = _primary_failure_message(errors)
    if primary:
        parts.append(primary)
    return '\n'.join(dict.fromkeys(parts))


def _summarise_runtime_failures(
    errors: list[str],
    deps: list[str],
    missing_helpers: list[str] | None = None,
) -> str:
    parts: list[str] = []
    missing_modules = _missing_runtime_modules(errors)
    if missing_modules:
        parts.append(f'Missing runtime module(s): {", ".join(missing_modules)}')
    helper_list = sorted(dict.fromkeys(missing_helpers or []))
    if helper_list:
        parts.append(
            f'Missing repo-local helper source/object(s): {", ".join(helper_list)}'
        )
    if deps:
        parts.append(f'Requires external dependency(s): {", ".join(deps)}')
    primary = _primary_failure_message(errors)
    if primary:
        parts.append(primary)
    return '\n'.join(dict.fromkeys(parts))

def compile_cobol(program: COBOLProgram, work_dir: str) -> tuple[str | None, str]:
    """
    Compile a .cbl file using GnuCOBOL.
    Returns (binary_path, error_message).
    binary_path is None on failure.
    """
    cobol_filepath = program.filepath
    src = Path(cobol_filepath).resolve()
    out = Path(work_dir) / src.stem
    cobc = config.GNUCOBOL_PATH or 'cobc'
    source_text = _read_source_text(cobol_filepath)
    prepared_src = _prepare_compile_source(cobol_filepath, source_text, work_dir)
    helper_objects, helper_errors, helper_missing = _compile_helper_objects(
        cobol_filepath, source_text, work_dir
    )
    compile_variants, deps, hinted_missing = _compile_variants(
        cobol_filepath, helper_objects
    )
    support_args = _compile_support_args(program, work_dir)
    errors: list[str] = list(helper_errors)
    missing_helpers = sorted(dict.fromkeys([*helper_missing, *hinted_missing]))
    for flags in compile_variants:
        try:
            result = subprocess.run(
                [cobc, *support_args, *flags, str(prepared_src), '-o', str(out)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(src.parent),
            )
            if result.returncode == 0:
                logger.info('Compiled %s -> %s using flags %s',
                            src.name, out.name, ' '.join(flags))
                return str(out), ''
            err = (result.stderr or result.stdout or 'compile error').strip()
            errors.append(f'{" ".join(flags)}: {err[:400]}')
        except subprocess.TimeoutExpired:
            errors.append(f'{" ".join(flags)}: Compilation timed out after 30s')
        except Exception as exc:
            errors.append(f'{" ".join(flags)}: {exc}')
    logger.warning('GnuCOBOL compile failed for %s: %s', src.name, errors[0][:200])
    return None, _summarise_compile_failures(errors, deps, missing_helpers)

def run_cobol_binary(
    binary_path: str,
    stdin_text: str,
    timeout: int = 10,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str, str]:
    """
    Execute a compiled COBOL binary, feeding stdin_text as input.
    Returns (stdout, stderr).
    Strips COBOL padding from numeric outputs.
    """
    try:
        result = subprocess.run(
            [binary_path],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return '', f'Runtime timed out after {timeout}s'
    except Exception as exc:
        return '', str(exc)


# ════════════════════════════════════════════════
# PHASE 2: GENERATE TEST INPUTS
# ════════════════════════════════════════════════

def _pic_max_value(pic: str) -> int:
    """Return the maximum integer value for a numeric PIC clause."""
    digits = re.findall(r'\d+', pic.replace('V', '').replace('9', ''))
    total = 0
    for p in pic.upper():
        if p == '9': total += 1
    for m in re.findall(r'9\((\d+)\)', pic.upper()):
        total += int(m) - 1  # already counted the 9 itself
    return 10 ** max(total, 1) - 1

def generate_test_inputs(program: COBOLProgram) -> list[dict]:
    """
    For each ACCEPT statement in the program, generate a grid of
    test input values covering:
      - zero / empty
      - a typical valid value
      - boundary (max) value
      - a negative or invalid value
    Returns a list of input dicts, one per test scenario.
    """
    # Find all variables read via ACCEPT
    accept_vars: list[str] = []
    for para in program.paragraphs:
        for stmt in para.statements:
            if stmt.verb.upper() == 'ACCEPT' and stmt.args:
                if re.search(r'\bFROM\s+(?:DATE|TIME)\b', stmt.raw, re.IGNORECASE):
                    continue
                accept_vars.append(stmt.args[0].upper().rstrip('.'))
    accept_vars = list(dict.fromkeys(accept_vars))  # preserve order, dedup

    if not accept_vars:
        logger.info('%s: no ACCEPT statements found, using single empty input',
                    program.program_id)
        return [{}]  # single test with no stdin

    # Build type map from data items
    type_map: dict[str, str] = {}
    pic_map: dict[str, str] = {}
    for item in program.all_data_items:
        type_map[item.name.upper()] = item.python_type
        if item.pic:
            pic_map[item.name.upper()] = item.pic.upper()

    # Generate value grids per variable
    decimal_separator = ',' if re.search(
        r'DECIMAL-POINT\s+IS\s+COMMA',
        program.raw_source,
        re.IGNORECASE,
    ) else '.'

    grids: list[list[str]] = []
    for var in accept_vars:
        py_type = type_map.get(var, 'str')
        pic = pic_map.get(var, '')
        if py_type == 'str':
            grids.append(['A', 'TEST', ' ' * 5])
        elif py_type == 'Decimal':
            grids.append([
                '0',
                f'1234{decimal_separator}56',
                f'99999{decimal_separator}99',
            ])
        else:  # int
            max_val = _pic_max_value(pic) if pic else 9999
            grids.append(['0', '100', str(min(max_val, 50000))])

    # Take the first value from each grid as scenario A, second as B, third as C
    # (full cartesian product would create too many cases for large programs)
    scenarios = []
    n = max(len(g) for g in grids)
    for i in range(n):
        scenario = {}
        for j, var in enumerate(accept_vars):
            scenario[var] = grids[j][min(i, len(grids[j]) - 1)]
        scenarios.append(scenario)
    return scenarios

def _inputs_to_stdin(inputs: dict) -> str:
    """Convert an inputs dict to a newline-separated stdin string."""
    return '\n'.join(str(v) for v in inputs.values()) + '\n'


# ════════════════════════════════════════════════
# PHASE 4: LLM EDGE-CASE AUGMENTATION
# ════════════════════════════════════════════════

_EDGE_CASE_PROMPT = '''
You are a COBOL testing expert. Given the business rules below,
suggest 3 additional edge-case input scenarios that would stress-test
boundary conditions and error handling.

Input variables accepted by the program: {accept_vars}
Business rules: {rules}

Return ONLY a JSON array of input dicts. Each dict maps
variable names (uppercase, hyphenated) to string values.
Example: [{"WS-BALANCE": "0", "WS-TRANSACTION-AMT": "50001"}]
'''

def augment_with_llm(
    program: COBOLProgram,
    business_rules: list[dict],
    existing_inputs: list[dict],
) -> list[dict]:
    """
    Ask Ollama to generate edge-case inputs based on business rules.
    Falls back to empty list if Ollama is unavailable.
    """
    accept_vars = []
    for para in program.paragraphs:
        for stmt in para.statements:
            if stmt.verb.upper() == 'ACCEPT' and stmt.args:
                if re.search(r'\bFROM\s+(?:DATE|TIME)\b', stmt.raw, re.IGNORECASE):
                    continue
                accept_vars.append(stmt.args[0].upper().rstrip('.'))
    if not accept_vars or not business_rules:
        return []

    rules_text = '\n'.join(f'- {r["rule"]}' for r in business_rules[:5])
    prompt = _EDGE_CASE_PROMPT.format(
        accept_vars=', '.join(accept_vars),
        rules=rules_text,
    )
    try:
        result = ollama_client.generate_json(
            model=config.MODEL_TESTGEN,
            prompt=prompt,
            fallback=[],
        )
        if isinstance(result, list):
            logger.info('LLM augmented %d edge cases for %s',
                        len(result), program.program_id)
            return result
    except OllamaError:
        logger.info('Ollama unavailable for edge-case augmentation — skipping')
    except Exception as exc:
        logger.warning('LLM augmentation failed: %s', exc)
    return []


# ════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════

def generate_oracle(
    program: COBOLProgram,
    business_rules: list[dict] | None = None,
    augment: bool = True,
    output_dir: str | None = None,
) -> BehavioralOracle:
    """
    Main Test Generation Agent entry point.

    1. Compile COBOL with GnuCOBOL
    2. Generate test input grid
    3. Run COBOL binary for each input -> capture stdout as ground truth
    4. (Optional) Augment with LLM edge cases

    If GnuCOBOL is not installed, returns a minimal oracle
    with compile_success=False so the pipeline can still run
    (validation will be skipped or marked PARTIAL).
    """
    config.setup()
    logger.info('=== Test Generation Agent: %s ===', program.program_id)

    source_text = program.raw_source or _read_source_text(program.filepath)
    preflight_issue = _oracle_preflight_issue(program, source_text)
    if preflight_issue:
        logger.info('%s: oracle skipped — %s', program.program_id, preflight_issue)
        return BehavioralOracle(
            program_id=program.program_id,
            cobol_filepath=program.filepath,
            compile_success=False,
            compile_error=preflight_issue,
        )

    if not _gnucobol_available():
        logger.warning('GnuCOBOL not found — returning empty oracle')
        return BehavioralOracle(
            program_id=program.program_id,
            cobol_filepath=program.filepath,
            compile_success=False,
            compile_error='GnuCOBOL (cobc) not installed',
        )

    work_dir = Path(config.TEMP_DIR) / program.program_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Compile
    binary_path, compile_err = compile_cobol(program, str(work_dir))
    if binary_path is None:
        return BehavioralOracle(
            program_id=program.program_id,
            cobol_filepath=program.filepath,
            compile_success=False,
            compile_error=compile_err,
        )

    compiled_modules = _prepare_called_modules(program, str(work_dir))
    runtime_env = os.environ.copy()
    if compiled_modules:
        work_dir_str = str(Path(work_dir).resolve())
        existing = runtime_env.get('COB_LIBRARY_PATH')
        runtime_env['COB_LIBRARY_PATH'] = (
            f'{work_dir_str}{os.pathsep}{existing}' if existing else work_dir_str
        )
    runtime_cwd = str(Path(program.filepath).resolve().parent)

    # Phase 2: Generate input scenarios
    oracle_inputs = generate_test_inputs(program)
    input_scenarios = list(oracle_inputs)
    deps = _detect_external_dependencies(source_text)
    missing_helpers = _missing_helper_artifacts(program.filepath, source_text)

    # Phase 4: LLM augmentation (before running, so we run all at once)
    if augment and business_rules:
        llm_inputs = augment_with_llm(program, business_rules, input_scenarios)
        input_scenarios = input_scenarios + llm_inputs

    # Phase 3: Run oracle
    test_cases: list[TestCase] = []
    runtime_errors: list[str] = []
    for i, inputs in enumerate(input_scenarios, 1):
        stdin = _inputs_to_stdin(inputs)
        stdout, stderr = run_cobol_binary(
            binary_path,
            stdin,
            cwd=runtime_cwd,
            env=runtime_env,
        )
        if stderr and not stdout:
            logger.warning('COBOL run %d failed: %s', i, stderr[:100])
            runtime_errors.append(stderr)
            continue
        src = 'oracle' if i <= len(oracle_inputs) else 'llm_augment'
        test_cases.append(TestCase(
            test_id=f'tc_{i:03d}',
            inputs=inputs,
            expected_output=stdout,
            description=', '.join(f'{k}={v}' for k, v in inputs.items()),
            source=src,
        ))

    if not test_cases and runtime_errors:
        return BehavioralOracle(
            program_id=program.program_id,
            cobol_filepath=program.filepath,
            compile_success=False,
            compile_error=_summarise_runtime_failures(
                runtime_errors, deps, missing_helpers
            ),
        )

    oracle = BehavioralOracle(
        program_id=program.program_id,
        cobol_filepath=program.filepath,
        test_cases=test_cases,
        compile_success=True,
    )
    logger.info('Oracle: %d test cases generated for %s',
               oracle.total_cases, program.program_id)

    # Save oracle to output dir
    out = Path(output_dir or config.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    oracle_path = out / f'{_oracle_basename(program)}_oracle.json'
    oracle_path.write_text(json.dumps(oracle.to_dict(), indent=2), encoding='utf-8')
    logger.info('Oracle written to %s', oracle_path)

    return oracle
