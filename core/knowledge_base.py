# ── FILE: cobol_moderniser/core/knowledge_base.py ─────────────────
"""
core/knowledge_base.py -- Persistent learning store.


Two-layer storage:
  SQLite  -- structured records (translations, failures, migration log)
  ChromaDB -- vector embeddings for semantic similarity search
             (lets Self-Repair Agent find similar past failures by meaning)
"""


from __future__ import annotations
import hashlib, json, logging, sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


import chromadb
try:
    from chromadb import Settings
except ImportError:  # pragma: no cover - compatibility with older ChromaDB
    from chromadb.config import Settings


from config import config


logger = logging.getLogger(__name__)




# ── Data models ─────────────────────────────────────────────────
@dataclass
class TranslationRecord:
    cobol_hash: str = ''         # SHA-256 of raw COBOL source
    program_id: str = ''
    cobol_code: str = ''
    translated_code: str = ''
    language: str = ''           # 'python' or 'java'
    success: bool = True
    accuracy_score: float = 1.0  # 0.0 - 1.0 equivalence
    method: str = ''
    confidence: float | None = None
    output_filepath: str = ''
    timestamp: str = ''
    def __post_init__(self):
        if not self.language:
            self.language = config.TARGET_LANGUAGE
        if self.confidence is not None:
            self.accuracy_score = self.confidence
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()




@dataclass
class FailureRecord:
    program_id: str = ''
    cobol_code: str = ''
    translated_code: str = ''
    error_message: str = ''
    diagnosis: str = ''          # LLM-generated root cause
    fix_applied: str = ''        # Fix that worked (empty if unresolved)
    resolved: bool = False
    error_type: str = ''
    diff_summary: str = ''
    timestamp: str = ''
    def __post_init__(self):
        if not self.error_message and self.diff_summary:
            self.error_message = self.diff_summary
        if not self.diagnosis and self.error_type:
            self.diagnosis = self.error_type
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()




@dataclass
class MigrationEvent:
    program_id: str
    module_path: str
    status: str  # started|translated|validated|failed|rolled_back
    notes: str = ''
    details: str = ''
    timestamp: str = ''
    def __post_init__(self):
        if not self.notes and self.details:
            self.notes = self.details
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()




# ── SQLite schema ───────────────────────────────────────────────
_SCHEMA = '''
CREATE TABLE IF NOT EXISTS translations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cobol_hash      TEXT UNIQUE NOT NULL,
    program_id      TEXT,
    cobol_code      TEXT,
    translated_code TEXT,
    output_filepath TEXT,
    language        TEXT DEFAULT 'python',
    success         INTEGER DEFAULT 1,
    accuracy_score  REAL DEFAULT 1.0,
    timestamp       TEXT
);
CREATE TABLE IF NOT EXISTS failures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    program_id      TEXT,
    cobol_code      TEXT,
    translated_code TEXT,
    error_message   TEXT,
    diagnosis       TEXT,
    fix_applied     TEXT,
    resolved        INTEGER DEFAULT 0,
    timestamp       TEXT
);
CREATE TABLE IF NOT EXISTS migration_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    program_id  TEXT,
    module_path TEXT,
    status      TEXT,
    notes       TEXT,
    timestamp   TEXT
);
CREATE INDEX IF NOT EXISTS idx_trans_hash ON translations(cobol_hash);
CREATE INDEX IF NOT EXISTS idx_fail_prog  ON failures(program_id);
'''




class KnowledgeBase:
    """
    Two-layer knowledge store:
      SQLite  -- translations, failures, migration log
      ChromaDB -- semantic search over failure messages + diagnoses
    """


    def __init__(self, db_path: str | None = None, chroma_path: str | None = None):
        self.db_path = db_path or config.DB_PATH
        self.chroma_path = chroma_path or config.CHROMA_PATH
        self._init_sqlite()
        self._init_chroma()


    # ── SQLite init ──────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        if self.db_path != ':memory:':
            Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        return conn


    def _init_sqlite(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            cols = {
                row['name']
                for row in conn.execute("PRAGMA table_info(translations)").fetchall()
            }
            if 'output_filepath' not in cols:
                conn.execute('ALTER TABLE translations ADD COLUMN output_filepath TEXT')
            if 'method' not in cols:
                conn.execute("ALTER TABLE translations ADD COLUMN method TEXT DEFAULT 'llm'")
            conn.execute("UPDATE translations SET method='hybrid' WHERE method IS NULL OR method=''")
        logger.debug('SQLite KB ready at %s', self.db_path)


    # ── ChromaDB init ────────────────────────────────────────────
    def _init_chroma(self):
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._failure_col = self._chroma.get_or_create_collection('failures')
        logger.debug('ChromaDB ready at %s', self.chroma_path)


    # ── Static helpers ───────────────────────────────────────────
    @staticmethod
    def hash_cobol(code: str) -> str:
        return hashlib.sha256(code.strip().encode()).hexdigest()


    # ── Translations (SQLite) ────────────────────────────────────
    def lookup_translation(self, cobol_code: str) -> Optional[TranslationRecord]:
        h = self.hash_cobol(cobol_code)
        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM translations WHERE cobol_hash=? AND success=1', (h,)
            ).fetchone()
        if not row:
            return None
        return TranslationRecord(
            cobol_hash=row['cobol_hash'], program_id=row['program_id'],
            cobol_code=row['cobol_code'], translated_code=row['translated_code'],
            output_filepath=row['output_filepath'] or '',
            language=row['language'], success=bool(row['success']),
            accuracy_score=row['accuracy_score'], timestamp=row['timestamp'])


    def save_translation(self, rec: TranslationRecord):
        with self._connect() as conn:
            conn.execute('''
                INSERT INTO translations
                  (cobol_hash,program_id,cobol_code,translated_code,output_filepath,
                   language,success,accuracy_score,method,timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(cobol_hash) DO UPDATE SET
                  success=excluded.success,
                  accuracy_score=excluded.accuracy_score,
                  translated_code=excluded.translated_code,
                  output_filepath=excluded.output_filepath,
                  method=excluded.method,
                  timestamp=excluded.timestamp''',
                (rec.cobol_hash, rec.program_id, rec.cobol_code[:8000],
                 rec.translated_code, rec.output_filepath,
                 rec.language, int(rec.success),
                 rec.accuracy_score, rec.method or 'llm', rec.timestamp))


    # ── Failures (SQLite + ChromaDB) ─────────────────────────────
    def save_failure(self, rec: FailureRecord) -> int:
        with self._connect() as conn:
            cur = conn.execute('''
                INSERT INTO failures
                  (program_id,cobol_code,translated_code,error_message,
                   diagnosis,fix_applied,resolved,timestamp)
                VALUES (?,?,?,?,?,?,?,?)''',
                (rec.program_id, rec.cobol_code[:5000], rec.translated_code[:5000],
                 rec.error_message, rec.diagnosis, rec.fix_applied,
                 int(rec.resolved), rec.timestamp))
            row_id = cur.lastrowid
        # Also index in ChromaDB for semantic search
        if rec.resolved and rec.diagnosis:
            try:
                doc = f'ERROR: {rec.error_message} DIAGNOSIS: {rec.diagnosis}'
                self._failure_col.upsert(
                    ids=[f'fail_{row_id}'],
                    documents=[doc],
                    metadatas=[{'program_id': rec.program_id,
                                'fix': rec.fix_applied[:500]}])
            except Exception as exc:
                logger.warning('ChromaDB upsert failed: %s', exc)
        return row_id


    def find_similar_failures(self, error_message: str, n: int = 5) -> list[dict]:
        """
        Semantic search: find past failures similar to this error message.
        Returns list of {program_id, fix} dicts from ChromaDB.
        """
        try:
            results = self._failure_col.query(
                query_texts=[error_message], n_results=n)
            metas = results.get('metadatas', [[]])[0]
            return metas if metas else []
        except Exception as exc:
            logger.warning('ChromaDB query failed: %s', exc)
            return []


    def get_recent_failures(self, program_id: str, limit: int = 5) -> list[dict]:
        """Get recent resolved failures for the same program (SQL)."""
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT error_message,diagnosis,fix_applied FROM failures '
                'WHERE program_id=? AND resolved=1 ORDER BY id DESC LIMIT ?',
                (program_id, limit)).fetchall()
        return [dict(r) for r in rows]


    def get_successful_patterns_for_program(
        self, program_id: str, n: int = 5
    ) -> list[dict]:
        """Return successful translation records from prior passes.
        Called by the Analyse phase to bias toward known-good strategies."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT program_id, cobol_hash, language,
                          accuracy_score,
                          COALESCE(method, 'llm') as method
                   FROM translations
                   WHERE success=1
                   ORDER BY accuracy_score DESC, id DESC
                   LIMIT ?""",
                (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def kb_informed_strategy(self, program_id: str) -> str:
        """Return 'RULE', 'LLM', or 'CACHE' recommendation
        based on what worked in prior passes for similar programs."""
        patterns = self.get_successful_patterns_for_program(program_id)
        if not patterns:
            return "LLM"
        methods = [p.get('method', 'llm') for p in patterns]
        rule_count = methods.count('rule')
        cache_count = methods.count('cache')
        total = len(methods)
        if cache_count > 0:
            return "CACHE"
        if rule_count >= total * 0.6:
            return "RULE"
        return "LLM"

    def get_learning_summary(self) -> dict:
        """Produce a structured learning summary across all passes.
        Used by LearningAgent and for paper §4 evidence."""
        stats = self.stats()
        with self._connect() as conn:
            cache_hits = conn.execute(
                "SELECT COUNT(*) FROM migration_log WHERE status='CACHE_HIT'"
            ).fetchone()[0]
            pass_count = conn.execute(
                "SELECT COUNT(DISTINCT notes) FROM migration_log WHERE module_path='PASS_MARKER'"
            ).fetchone()[0]
        return {
            "total_translations_cached": stats['translations'],
            "successful_translations": stats['successful_translations'],
            "failure_remediations": stats['resolved_failures'],
            "semantic_index_size": stats['chroma_embeddings'],
            "cache_hits_across_passes": cache_hits,
            "passes_completed": pass_count,
            "self_repair_rate": round(
                stats['resolved_failures'] / stats['failures'], 3
            ) if stats['failures'] > 0 else 0.0,
            "interpretation": (
                f"KB contains {stats['translations']} cached translations. "
                f"Pass N+1 may find {stats['successful_translations']} "
                f"successful patterns before invoking LLM. "
                f"Semantic failure index has {stats['chroma_embeddings']} entries."
            )
        }

    def mark_pass_start(self, pass_number: int, repos: list[str]):
        """Record start of a new pipeline pass for multi-pass tracking."""
        import json as _json
        self.log_event(MigrationEvent(
            program_id="__PIPELINE__",
            module_path="PASS_MARKER",
            status=f"PASS_{pass_number}_START",
            notes=_json.dumps({"pass": pass_number, "repos": repos}),
        ))

    def record_cache_hit(self, program_id: str, accuracy_score: float):
        """Record when a translation is served from KB cache (evidence of learning)."""
        self.log_event(MigrationEvent(
            program_id=program_id,
            module_path="CACHE_HIT",
            status="CACHE_HIT",
            notes=f"accuracy={accuracy_score:.3f}",
        ))


    # ── Migration log ────────────────────────────────────────────
    def log_event(self, event: MigrationEvent):
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO migration_log(program_id,module_path,status,notes,timestamp)'
                ' VALUES(?,?,?,?,?)',
                (event.program_id, event.module_path, event.status,
                 event.notes, event.timestamp))


    # ── Stats ────────────────────────────────────────────────────
    def stats(self) -> dict:
        with self._connect() as conn:
            total_t = conn.execute('SELECT COUNT(*) FROM translations').fetchone()[0]
            succ_t  = conn.execute('SELECT COUNT(*) FROM translations WHERE success=1').fetchone()[0]
            total_f = conn.execute('SELECT COUNT(*) FROM failures').fetchone()[0]
            res_f   = conn.execute('SELECT COUNT(*) FROM failures WHERE resolved=1').fetchone()[0]
            total_e = conn.execute('SELECT COUNT(*) FROM migration_log').fetchone()[0]
        chroma_count = self._failure_col.count()
        return {
            'translations': total_t,
            'failures': total_f,
            'failures_resolved': res_f,
            'events': total_e,
            'chroma_embeddings': chroma_count,
            'total_translations': total_t,
            'successful_translations': succ_t,
            'total_failures': total_f,
            'resolved_failures': res_f,
            'chroma_indexed_failures': chroma_count,
        }
