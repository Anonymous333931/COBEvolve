"""
main.py -- CLI entry point for the Self-Evolving COBOL Modernisation Pipeline.

100% Open-Source. Requires Ollama running locally.

Usage
-----
# Analyse + translate an entire repository (repo-wise)
python main.py run --source ./samples/bank_system

# Analyse multiple repos together
python main.py run --source ./samples/bank_system --source ./samples/payroll_system

# Analyse only (no translation)
python main.py plan --source ./samples/bank_system

# Translate a single file
python main.py translate --file ./samples/bank_system/ACCTPROC.cbl

# Show knowledge base stats
python main.py stats

# Check if Ollama is running and models are available
python main.py check
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

from config import config

console = Console()


# ════════════════════════════════════════════════════════════════
# CLI group
# ════════════════════════════════════════════════════════════════

@click.group()
@click.option("--db", default=None, help="SQLite KB path (default: cobol_evolution.db)")
@click.option("--output", default=None, help="Output directory (default: ./modernised_output)")
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging")
@click.pass_context
def cli(ctx, db, output, verbose):
    """Self-Evolving COBOL Modernisation Pipeline — 100% Open-Source."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db or config.DB_PATH
    ctx.obj["output"] = output or config.OUTPUT_DIR
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )


# ════════════════════════════════════════════════════════════════
# check — verify environment
# ════════════════════════════════════════════════════════════════

@cli.command()
def check():
    """Check environment: Ollama server, models, GnuCOBOL."""
    import shutil
    import urllib.request

    console.print(Panel("[bold cyan]Environment Check[/]", expand=False))

    # Ollama server
    try:
        urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        console.print(f"  [green]✓[/] Ollama server reachable at {config.OLLAMA_BASE_URL}")
    except Exception:
        console.print(f"  [red]✗[/] Ollama server NOT reachable at {config.OLLAMA_BASE_URL}")
        console.print("     Run: [bold]ollama serve[/]")

    # Models
    import ollama as ol
    try:
        client = ol.Client(host=config.OLLAMA_BASE_URL)
        pulled = {
            name
            for model in client.list().get("models", [])
            for name in (
                model.get("name", ""),
                model.get("model", ""),
            )
            if name
        }
        pulled_aliases = pulled | {
            name.removesuffix(":latest") for name in pulled
        }
        for model in [
            config.MODEL_TRANSLATION,
            config.MODEL_ANALYSIS,
            config.MODEL_REPAIR,
            config.MODEL_TESTGEN,
        ]:
            if model in pulled_aliases:
                console.print(f"  [green]✓[/] Model {model}")
            else:
                console.print(f"  [yellow]?[/] Model {model} not found")
                console.print(f"     Pull with: [bold]ollama pull {model}[/]")
    except Exception as exc:
        console.print(f"  [red]✗[/] Cannot list Ollama models: {exc}")

    # GnuCOBOL
    cobc = shutil.which("cobc")
    if cobc:
        console.print(f"  [green]✓[/] GnuCOBOL found at {cobc}")
    else:
        console.print("  [yellow]![/] GnuCOBOL (cobc) not found — oracle tests disabled")
        console.print("     Install: [bold]sudo apt install gnucobol[/]  (Ubuntu/Debian)")

    # Python packages
    pkgs = ["crewai", "chromadb", "networkx", "ollama", "click", "rich"]
    for pkg in pkgs:
        try:
            __import__(pkg)
            console.print(f"  [green]✓[/] Python package: {pkg}")
        except ImportError:
            console.print(f"  [red]✗[/] Missing package: {pkg}")
            console.print(f"     Install: [bold]pip install {pkg}[/]")


# ════════════════════════════════════════════════════════════════
# run — full pipeline
# ════════════════════════════════════════════════════════════════

@cli.command()
@click.option(
    "--source", "-s", multiple=True, required=True,
    help="Path to COBOL file or directory (repo). Repeat for multiple repos.",
)
@click.option("--no-rules", is_flag=True, help="Skip Ollama business-rule extraction.")
@click.option("--no-llm", is_flag=True, help="Disable LLM translation and self-repair.")
@click.option("--max-modules", default=None, type=int, help="Limit modules processed.")
@click.option("--crewai", "use_crewai", is_flag=True, help="Use CrewAI orchestration.")
@click.pass_context
def run(ctx, source, no_rules, no_llm, max_modules, use_crewai):
    """
    Run the full MAPE-K pipeline.

    Examples:

      # Single repo
      python main.py run --source ./samples/bank_system

      # Multiple repos (repo-wise)
      python main.py run -s ./samples/bank_system -s ./samples/payroll_system

      # Fast mode (no Ollama rule extraction)
      python main.py run --source ./samples/bank_system --no-rules
    """
    from core.orchestrator import Orchestrator

    sources = list(source)
    console.print(Panel(
        f"[bold green]Self-Evolving COBOL Pipeline[/]\n"
        f"Source(s) : {', '.join(sources)}\n"
        f"Output    : {ctx.obj['output']}\n"
        f"Mode      : {'CrewAI' if use_crewai else 'Direct'}\n"
        f"Rules     : {'disabled' if no_rules else 'enabled (Ollama)'}\n"
        f"LLM       : {'disabled' if no_llm else 'enabled'}",
        title="Starting",
    ))

    orch = Orchestrator(db_path=ctx.obj["db"], output_dir=ctx.obj["output"])

    if len(sources) == 1:
        report = orch.run(
            sources[0],
            extract_rules=not no_rules,
            max_modules=max_modules,
            use_crewai=use_crewai,
            use_llm=not no_llm,
        )
    else:
        report = orch.run_repos(
            sources,
            extract_rules=not no_rules,
            max_modules=max_modules,
            use_llm=not no_llm,
        )

    # Rich results table
    table = Table(title="Module Results", show_lines=True)
    table.add_column("Priority", style="dim", width=8)
    table.add_column("Program", style="cyan")
    table.add_column("Refactor")
    table.add_column("Translate")
    table.add_column("Validate")
    table.add_column("Repair")
    table.add_column("Learn")
    table.add_column("Pass %", justify="right")
    table.add_column("Time", justify="right")

    status_color = {
        "PASS": "green", "FIXED": "green",
        "PARTIAL": "yellow", "ALREADY_PASSING": "green",
        "FAIL": "red", "ROLLBACK": "red bold",
        "SKIPPED": "dim", "FAILED": "red",
        "ANALYZED": "green", "NORMALIZED": "green",
        "RECORDED": "green", "PENDING": "yellow",
    }

    for o in report.outcomes:
        tc = status_color.get(o.translation_status, "white")
        vc = status_color.get(o.validation_status, "white")
        rc = status_color.get(o.repair_status, "white")
        fc = status_color.get(o.refactor_status, "white")
        lc = status_color.get(o.learning_status, "white")
        table.add_row(
            str(o.cycle),
            o.program_id,
            f"[{fc}]{o.refactor_status}[/]",
            f"[{tc}]{o.translation_status}[/]",
            f"[{vc}]{o.validation_status}[/]",
            f"[{rc}]{o.repair_status}[/]",
            f"[{lc}]{o.learning_status}[/]",
            f"{o.pass_rate:.0%}",
            f"{o.elapsed_seconds:.1f}s",
        )

    console.print(table)
    console.print(
        f"\n[bold green]Automation rate: {report.automation_rate:.1%}[/]  "
        f"({report.auto_modernised}/{report.total_modules} modules fully modernised)"
    )


# ════════════════════════════════════════════════════════════════
# plan — analyse and show plan only
# ════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--source", "-s", required=True, help="COBOL directory or file.")
@click.option("--no-rules", is_flag=True)
@click.pass_context
def plan(ctx, source, no_rules):
    """Analyse codebase and show migration plan (no translation)."""
    from core.orchestrator import Orchestrator
    from agents.planning_agent import summarise_plan

    orch = Orchestrator(db_path=ctx.obj["db"], output_dir=ctx.obj["output"])
    kg, graph = orch.monitor(source, extract_rules=not no_rules)
    mp = orch.analyze(kg, graph)
    console.print(summarise_plan(mp))


# ════════════════════════════════════════════════════════════════
# translate — single file
# ════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--file", "-f", "filepath", required=True, help="Path to .cbl file.")
@click.option("--validate", "do_validate", is_flag=True, help="Also run validation.")
@click.option("--no-cache", is_flag=True, help="Skip KB cache.")
@click.pass_context
def translate(ctx, filepath, do_validate, no_cache):
    """Translate a single COBOL file to Python."""
    from core.knowledge_base import KnowledgeBase
    from agents.translation_agent import translate_module, TranslationResult
    from agents.planning_agent import ScoredModule
    from agents.test_generation_agent import generate_oracle
    from agents.validation_agent import validate as run_validate, format_validation_report
    from utils.cobol_parser import parse_cobol_file

    config.setup()
    kb = KnowledgeBase(db_path=ctx.obj["db"])
    program = parse_cobol_file(filepath)

    # Minimal ScoredModule for translate_module()
    module = ScoredModule(
        program_id=program.program_id,
        filepath=filepath,
        priority=1,
    )

    console.print(f"[cyan]Translating {program.program_id} ...[/]")
    tr = translate_module(module, program, kb, output_dir=ctx.obj["output"])

    console.print(Syntax(tr.translated_code, "python", line_numbers=True,
                         theme="monokai"))

    if do_validate:
        console.print("[cyan]Generating oracle tests ...[/]")
        oracle = generate_oracle(program, output_dir=ctx.obj["output"])
        vr = run_validate(tr, oracle, cobol_source=program.raw_source)
        console.print(format_validation_report(vr))

    console.print(f"[green]Saved → {tr.output_filepath}[/]")


# ════════════════════════════════════════════════════════════════
# stats — knowledge base statistics
# ════════════════════════════════════════════════════════════════

@cli.command()
@click.pass_context
def stats(ctx):
    """Show knowledge base statistics."""
    from core.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(db_path=ctx.obj["db"])
    s = kb.stats()

    table = Table(title="Knowledge Base Statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Translations stored", str(s["translations"]))
    table.add_row("Failures logged", str(s["failures"]))
    table.add_row("Failures resolved", str(s["failures_resolved"]))
    table.add_row("Events logged", str(s["events"]))
    table.add_row("ChromaDB embeddings", str(s["chroma_embeddings"]))

    if s["failures"] > 0:
        rate = s["failures_resolved"] / s["failures"]
        table.add_row("Self-repair success rate", f"{rate:.1%}")

    console.print(table)


# ════════════════════════════════════════════════════════════════
# report — show the latest pipeline report
# ════════════════════════════════════════════════════════════════

@cli.command()
@click.option(
    "--file",
    "report_path",
    default=None,
    help="Path to pipeline_report.json (default: <output>/pipeline_report.json).",
)
@click.pass_context
def report(ctx, report_path):
    """Display a saved pipeline report."""
    default_path = Path(ctx.obj["output"]) / "pipeline_report.json"
    path = Path(report_path or default_path)
    if not path.exists():
        raise click.ClickException(f"Pipeline report not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    console.print(Panel.fit(f"[bold cyan]Pipeline Report[/]\n{path}"))

    summary = Table(title="Summary")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Total modules", str(data.get("total_modules", 0)))
    summary.add_row("Auto-modernised", str(data.get("auto_modernised", 0)))
    summary.add_row("Partial", str(data.get("partially_modernised", 0)))
    summary.add_row("Rolled back", str(data.get("rolled_back", 0)))
    summary.add_row("Failed", str(data.get("failed", 0)))
    summary.add_row("Skipped", str(data.get("skipped", 0)))
    summary.add_row("Automation rate", f"{float(data.get('automation_rate', 0.0)):.1%}")
    summary.add_row("Elapsed", f"{float(data.get('elapsed_seconds', 0.0)):.1f}s")
    console.print(summary)

    outcomes = data.get("outcomes", [])
    if outcomes:
        table = Table(title="Outcomes", show_lines=True)
        table.add_column("Cycle", style="dim", width=6)
        table.add_column("Program", style="cyan")
        table.add_column("Refactor")
        table.add_column("Translate")
        table.add_column("Validate")
        table.add_column("Repair")
        table.add_column("Learn")
        table.add_column("Pass %", justify="right")

        for outcome in outcomes:
            table.add_row(
                str(outcome.get("cycle", "")),
                outcome.get("program_id", ""),
                outcome.get("refactor_status", ""),
                outcome.get("translation_status", ""),
                outcome.get("validation_status", ""),
                outcome.get("repair_status", ""),
                outcome.get("learning_status", ""),
                f"{float(outcome.get('pass_rate', 0.0)):.0%}",
            )
        console.print(table)


# ════════════════════════════════════════════════════════════════
# comprehend — show knowledge graph only
# ════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--source", "-s", required=True)
@click.option("--save", default=None, help="Save knowledge graph JSON to this path.")
@click.option("--no-rules", is_flag=True)
def comprehend(source, save, no_rules):
    """Run the Comprehension Agent and display the knowledge graph."""
    from agents.comprehension_agent import comprehend as run_comprehend
    from agents.comprehension_agent import summarise_knowledge_graph

    config.setup()
    kg, graph = run_comprehend(source, extract_rules=not no_rules)
    console.print(summarise_knowledge_graph(kg))

    if save:
        Path(save).write_text(json.dumps(kg, indent=2), encoding="utf-8")
        console.print(f"[green]Knowledge graph saved → {save}[/]")


@cli.command()
@click.option("--db", "db_path", default=None, help="Path to evolution SQLite DB.")
@click.option(
    "--output", "--out", "out_path", default="paper_evidence.json",
    help="Where to write the evidence JSON."
)
@click.pass_context
def evidence(ctx, db_path, out_path):
    """Generate paper §4 evidence from KB — shows self-evolution across passes."""
    from core.knowledge_base import KnowledgeBase
    from agents.learning_agent import summarise_learning_across_passes

    kb = KnowledgeBase(db_path=db_path or ctx.obj["db"])
    summary = summarise_learning_across_passes(kb)
    learning = kb.get_learning_summary()
    combined = {**summary, **learning}

    Path(out_path).write_text(json.dumps(combined, indent=2), encoding="utf-8")

    table = Table(title="Self-Evolution Evidence (Paper §4)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total translations in KB", str(combined.get("total_translations_cached", 0)))
    table.add_row("Successful patterns", str(combined.get("successful_translations", 0)))
    table.add_row("Cache hits across passes", str(combined.get("cache_hits_across_passes", 0)))
    table.add_row("Semantic failure embeddings", str(combined.get("semantic_index_size", 0)))
    table.add_row("Failure remediations", str(combined.get("failure_remediations", 0)))
    table.add_row("Self-repair rate", f"{combined.get('self_repair_rate', 0):.1%}")
    table.add_row("Passes completed", str(combined.get("passes_completed", 0)))
    console.print(table)
    console.print(f"\n[green]Evidence JSON → {out_path}[/]")
    console.print(f"\n[dim]{combined.get('interpretation', '')}[/]")


if __name__ == "__main__":
    cli()
