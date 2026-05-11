# COBEvolve: Towards Self-Evolving Legacy COBOL Systems via Multi-Agent AI

> Companion artifact for the paper submitted to ASE 2026 NIER Track.

---

## What This Is

Most COBOL modernisation tools translate a codebase once and move on. COBEvolve takes a different approach: it treats modernisation as a continuous, multi-pass process where the system learns from each run and gets better at the next one. The core idea is a persistent knowledge base that accumulates translation outcomes, failure records, and agent decisions across passes, so the second run is always cheaper than the first.

The architecture is built around six specialised agents coordinated by a MAPE-K (Monitor, Analyse, Plan, Execute, Knowledge) control loop. Each agent wraps previously published tools as its backend — the pipeline is glue code connecting existing work, not a new tool built from scratch.

The paper reports results from running COBEvolve over 14 X-COBOL repositories (544 COBOL programs, 3 passes). Knowledge base cache reuse went from 9% in Pass 1 to 100% in Passes 2 and 3, with an average translation accuracy of 0.92 across oracle test cases.

---

## Repository Structure

```
cobol_moderniser/
│
├── main.py                    # CLI entry point (run, plan, translate, stats, check)
├── run_cobevolve.py           # Full multi-pass production runner
├── show_stats.py              # Statistics of 14 repo- 3 Pass Pipeline results
├── config.py                  # Centralised configuration with env var overrides
├── requirements.txt           # Python dependencies
│
├── agents/
│   ├── comprehension_agent.py # Parses COBOL, builds dependency graph, extracts rules
│   ├── refactoring_agent.py   # Identifies dead code, RVF scoring
│   ├── translation_agent.py   # 3-layer: KB cache → rule engine → LLM
│   ├── test_generation_agent.py # GnuCOBOL oracle compilation, I/O capture
│   ├── validation_agent.py    # Behavioural equivalence checking, pass rate
│   ├── learning_agent.py      # Writes outcomes to KB after each module
│   ├── planning_agent.py      # MAPE-K Analyse phase, RVF-based ordering
│   └── self_repair_agent.py   # LLM-assisted repair on FAIL/PARTIAL verdicts
│
├── core/
│   ├── orchestrator.py        # MAPE-K loop coordinator, CrewAI integration
│   └── knowledge_base.py      # SQLite + ChromaDB dual-layer storage
│
├── utils/
│   ├── cobol_parser.py        # COBOL file parser (all 4 divisions)
│   ├── ollama_client.py       # Retry-safe HTTP wrapper for Ollama
│   └── graph_utils.py         # NetworkX dependency graph helpers
│
├── samples/
│   └── X-COBOL_files/         # 14 repositories, 544 COBOL programs, 109 copybooks
│       ├── GaloisGirl_Coding/          (39 files)
│       ├── IBM_example-health-apis/    (29 files, 9 copybooks)
│       ├── Martinfx_Cobol/             (28 files)
│       ├── abrignoli_COBSOFT/          (45 files, 72 copybooks)
│       ├── bhbandam_AZ-Legacy-Engineering/ (21 files)
│       ├── bmcsoftware_vscode-ispw/    (35 files, 23 copybooks)
│       ├── debinix_openjensen/         (29 files, 3 copybooks)
│       ├── gbeine_COBOLUnit/           (48 files)
│       ├── lucasrmagalhaes_learning-COBOL/ (42 files)
│       ├── neopragma_cobol-unit-test/  (24 files)
│       ├── seanpm2001_SNU_2D_ProgrammingTools_IDE_COBOL/ (20 files, 2 copybooks)
│       ├── thospfuller_rcoboldi/       (20 files)
│       ├── ve3wwg_cobcurses/           (86 files)
│       └── z390development_z390/       (78 files)
│
├── tests/
│   ├── test_step1.py              # KB + parser unit tests
│   ├── test_step2.py              # ComprehensionAgent tests
│   ├── test_step3.py              # PlanningAgent tests
│   ├── test_cobevolve_agents.py   # RefactoringAgent + LearningAgent tests
│   └── test_bug_report_fixes.py   # Regression tests
│
├── cobevolve_full_run.db          # SQLite DB from the production 3-pass run
├── chroma_db/                     # ChromaDB vector store (7 failure embeddings)
├── cobevolve_architecture.svg     # Architecture diagram (source)
└── full_run_output/
    └── pass1/ # pass 1 results
    └── pass2/ # pass 2 results
    └── pass3/ # pass 3 results
    └── reports/
        └── dataset_inventory.json # Pre-run dataset summary
        └── evolution_evidence.json # stats of each pass are stored
        └── final_summary.md # summary after execution of pipeline 
```

---

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) running locally with `llama3.2` pulled
- GnuCOBOL (`cobc`) installed for oracle compilation (optional — pipeline degrades gracefully without it)

Install GnuCOBOL on Ubuntu/Debian:
```bash
sudo apt-get install gnucobol
```

Install GnuCOBOL on macOS:
```bash
brew install gnu-cobol
```

---

## Installation

```bash
# Clone or extract the project
cd COBEvolve

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull llama3.2
```

---

## Quick Start

### Check that Ollama is running and the model is available
```bash
python main.py check
```

### Translate a single COBOL file
```bash
python main.py translate --file samples/X-COBOL_files/z390development_z390/TESTFIL2.CBL
```

### Analyse a repository (planning only, no translation)
```bash
python main.py plan --source samples/X-COBOL_files/GaloisGirl_Coding
```

### Run the full modernisation pipeline on one repository
```bash
python main.py run --source samples/X-COBOL_files/GaloisGirl_Coding
```

### Run on multiple repositories
```bash
python main.py run \
  --source samples/X-COBOL_files/GaloisGirl_Coding \
  --source samples/X-COBOL_files/Martinfx_Cobol
```

### View knowledge base statistics
```bash
python main.py stats
```

---

## Full Multi-Pass Run (Production)

`run_cobevolve.py` is the entrypoint used in the paper. It runs all 14 X-COBOL repositories through 3 complete passes, writing results to a dedicated SQLite database and output directory.

```bash
# Full 3-pass run on all 14 repos
python run_cobevolve.py --dataset-root samples/X-COBOL_files --passes 3

# Resume an interrupted run from pass 2
python run_cobevolve.py --dataset-root samples/X-COBOL_files --passes 3 --resume-from 2

# Custom DB path and output directory
python run_cobevolve.py \
    --dataset-root samples/X-COBOL_files \
    --passes 3 \
    --db my_run.db \
    --output-dir my_run_output
```

Translated Python files land in `full_run_output/pass_1/`, `pass_2/`, `pass_3/`. The migration log, translation records, and failure records are written to `cobevolve_full_run.db`.

---

## Viewing Run Statistics

After a completed run, query the database directly:

```bash
python3 show_stats.py
```

Or query manually:

```python
import sqlite3

conn = sqlite3.connect('cobevolve_full_run.db')
cur = conn.cursor()

# Overall counts
cur.execute('SELECT COUNT(*) FROM translations')
print('Translations stored:', cur.fetchone()[0])

cur.execute('SELECT COUNT(*) FROM failures')
print('Failures logged:', cur.fetchone()[0])

# Cache hit rate per pass
cur.execute("SELECT COUNT(*) FROM migration_log WHERE status='CACHE_HIT'")
print('Total cache hits:', cur.fetchone()[0])

conn.close()
```

The ChromaDB vector store is at `chroma_db/`. To inspect it:

```python
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
col = client.get_collection('cobol_failures')
print('Failure embeddings:', col.count())
```

---

## Configuration

All settings live in `config.py` and can be overridden via environment variables or a `.env` file in the project root.

| Environment Variable     | Default                      | Description                            |
|--------------------------|------------------------------|----------------------------------------|
| `COBOL_DB_PATH`          | `cobol_evolution.db`         | SQLite database path                   |
| `COBOL_OUTPUT_DIR`       | `./modernised_output`        | Where translated Python files are saved |
| `COBOL_CHROMA_PATH`      | `./chroma_db`                | ChromaDB vector store directory        |
| `OLLAMA_BASE_URL`        | `http://127.0.0.1:11434`     | Ollama server URL                      |
| `GNUCOBOL_PATH`          | `/usr/bin/cobc`              | Path to GnuCOBOL compiler              |
| `TARGET_LANGUAGE`        | `python`                     | Translation target (`python` or `java`)|
| `MODEL_TRANSLATION`      | `llama3.2`                   | LLM model for translation              |
| `MODEL_ANALYSIS`         | `llama3.2`                   | LLM model for comprehension            |
| `MODEL_REPAIR`           | `llama3.2`                   | LLM model for self-repair              |
| `MODEL_TESTGEN`          | `llama3.2`                   | LLM model for test generation          |
| `VALIDATION_THRESHOLD`   | `0.8`                        | Pass rate threshold for PASS verdict   |
| `MAX_REPAIR_RETRIES`     | `3`                          | Max LLM repair attempts before rollback|

Example `.env` file:
```
OLLAMA_BASE_URL=http://127.0.0.1:11434
TARGET_LANGUAGE=python
VALIDATION_THRESHOLD=0.8
GNUCOBOL_PATH=/usr/local/bin/cobc
```

---

## How the Architecture Works

COBEvolve applies the MAPE-K autonomic computing loop to COBOL modernisation:

```
MONITOR   →  ComprehensionAgent scans repo, builds dependency graph,
             flags modules with known KB failures
             
ANALYSE   →  PlanningAgent queries KB for each module.
             Cache hit → route directly to Translation (no LLM needed).
             New module → full pipeline.
             
PLAN      →  Modules ordered by RVF (Risk/Value/Feasibility) score.
             MigrationPlan serialised for execution.
             
EXECUTE   →  TranslationAgent (3-layer: cache → rules → LLM)
             → TestGenerationAgent (GnuCOBOL oracle)
             → ValidationAgent (pass rate check)
             → SelfRepairAgent (if FAIL/PARTIAL)
             
KNOWLEDGE →  LearningAgent writes translation record, validation
             verdict, repair outcome to SQLite + ChromaDB.
             Next pass reads this back.
```

The self-evolution mechanism is simple: the TranslationAgent hashes the incoming COBOL source and checks whether that hash already exists in the `translations` table. If it does, the stored Python is returned immediately — no LLM call, no oracle compilation, no test execution. This is why Pass 2 and Pass 3 show 100% cache reuse once the KB is built from Pass 1.

---

## The Six Agents

| Agent | Lines | What it does | Grounded in |
|---|---|---|---|
| ComprehensionAgent | 476 | Parses COBOL, builds NetworkX graph, extracts business rules | COBREX (ICSME 2022), A-COBREX (ICSE 2025) |
| RefactoringAgent | 187 | Dead code detection, paragraph consolidation, RVF scoring | COBSmell |
| TranslationAgent | 2,554 | KB cache → rule engine → LLM | COB2PY (ICSME 2025) |
| TestGenerationAgent | 1,151 | GnuCOBOL oracle compilation, I/O capture | COBMaker (ICPC 2026) |
| ValidationAgent | 465 | Behavioural equivalence checking, repair trigger | — |
| LearningAgent | 131 | Writes all outcomes to KB | MAPE-K Knowledge component |
| PlanningAgent | 394 | RVF scoring, migration plan (MAPE-K Analyse) | — |
| SelfRepairAgent | 611 | LLM-assisted repair, up to 3 retries | — |

---

## Running the Tests

```bash
# All tests (Ollama-dependent tests auto-skip if Ollama is offline)
pytest tests/ -v

# Just the unit tests that do not need Ollama
pytest tests/test_step1.py tests/test_step3.py -v

# Regression tests
pytest tests/test_bug_report_fixes.py -v
```

---

## Pre-computed Run Results

The repository ships with results from the full 3-pass production run described in the paper:

- `cobevolve_full_run.db` — SQLite database with 10,383 migration log events, 543 translation records, 87 failure records
- `chroma_db/` — ChromaDB with 7 semantic failure embeddings
- `full_run_output/reports/dataset_inventory.json` — pre-run dataset summary
- `full_run_output/reports/evolution_evidence.json` — stats of each pass are stored
- `full_run_output/reports/final_summary.md` — summary after execution of pipeline 

These are included so reviewers can inspect the raw data without re-running the pipeline (which requires Ollama and takes several hours).

---

## Dataset: X-COBOL

The evaluation uses the X-COBOL dataset (Ali et al., arXiv:2306.04892), a collection of real COBOL repositories sourced from GitHub. The 14 repositories included here cover diverse domains:

| Repository | Domain | Files | Copybooks |
|---|---|---|---|
| GaloisGirl_Coding | Competitive programming | 39 | 0 |
| IBM_example-health-apis | Healthcare APIs (CICS) | 29 | 9 |
| Martinfx_Cobol | General examples | 28 | 0 |
| abrignoli_COBSOFT | Business software | 45 | 72 |
| bhbandam_AZ-Legacy-Engineering | Banking (AZ legacy) | 21 | 0 |
| bmcsoftware_vscode-ispw | BMC ISPW examples | 35 | 23 |
| debinix_openjensen | CGI web programming | 29 | 3 |
| gbeine_COBOLUnit | Unit testing framework | 48 | 0 |
| lucasrmagalhaes_learning-COBOL | Learning examples | 42 | 0 |
| neopragma_cobol-unit-test | Unit testing examples | 24 | 0 |
| seanpm2001_SNU_2D_ProgrammingTools | IDE examples | 20 | 2 |
| thospfuller_rcoboldi | rCOBOLdi parser tests | 20 | 0 |
| ve3wwg_cobcurses | Curses terminal library | 86 | 0 |
| z390development_z390 | z390 emulator suite | 78 | 0 |
| **Total** | | **544** | **109** |

---

## Known Limitations

- **Oracle compilation**: GnuCOBOL cannot compile CICS-dependent programs, copybooks used as includes, or programs with DB2 embedded SQL. These are skipped with a logged reason. About 40% of the dataset falls into this category.
- **Self-repair**: The SelfRepairAgent resolved only 7 of 87 failures in the paper run. It is functional but not yet robust enough to handle the range of failure types encountered.
- **Cache reuse**: 100% cache reuse in Passes 2–3 demonstrates that the KB works, but it does not yet test generalisation to COBOL constructs not seen in Pass 1. That requires held-out evaluation not included here.
- **Local LLM**: The pipeline uses Ollama/llama3.2 locally. Results may differ with larger or cloud-hosted models.

---

## Paper Reference

If you use this code or dataset in your research, please cite:

```
Anonymous Author(s). 2026. COBEvolve: Towards Self-Evolving Legacy COBOL Systems
via Multi-Agent AI. In ASE '26: 41st IEEE/ACM International Conference on
Automated Software Engineering. ACM.
```

---

## AI Disclosure

AI tools were used in developing the COBEvolve prototype and in drafting portions
of the accompanying paper. All experimental results, claims, and interpretations
are the authors' own responsibility.
