"""Typer CLI interface for clinical note generation, arbitration, validation, and reporting."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arbiter.generator import NoteGenerator
from arbiter.judge import ClinicalJudge
from arbiter.scorer import SingleNoteScorer
from arbiter.stats import generate_validation_summary

app = typer.Typer(
    name="arbiter",
    help="Clinical Note Arbitration CLI: Multi-LLM clinical documentation judge and evaluation harness.",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    transcript_file: Optional[Path] = typer.Option(
        None,
        "--transcript-file",
        "-f",
        help="Path to text file containing doctor-patient dialogue.",
    ),
    transcript_text: Optional[str] = typer.Option(
        None, "--text", "-t", help="Raw dialogue transcript string."
    ),
    model_a: str = typer.Option("mock-generator", "--model-a", help="LLM identifier for Note A."),
    model_b: str = typer.Option("mock-generator", "--model-b", help="LLM identifier for Note B."),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional output path to save generated candidate notes."
    ),
):
    """Generate two competing clinical SOAP notes from a consultation transcript."""
    if transcript_file and transcript_file.exists():
        transcript = transcript_file.read_text(encoding="utf-8")
    elif transcript_text:
        transcript = transcript_text
    else:
        # Default to bundled sample encounter
        sample_path = (
            Path(__file__).resolve().parent.parent / "data" / "samples" / "aci_bench_samples.json"
        )
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
            transcript = samples[0]["transcript"]
            rprint(
                "[yellow]No transcript provided. Using default ACI-Bench sample encounter.[/yellow]"
            )
        else:
            rprint("[red]Error: Please provide a transcript via --transcript-file or --text.[/red]")
            raise typer.Exit(code=1)

    rprint(
        Panel(
            f"[bold cyan]Generating Clinical Notes with Models:[/bold cyan]\n- Note A: {model_a}\n- Note B: {model_b}",
            title="Clinical Note Generation",
        )
    )

    gen_a = NoteGenerator(model_name=model_a)
    gen_b = NoteGenerator(model_name=model_b)

    with console.status("[bold green]Generating Note A..."):
        note_a = gen_a.generate_note(transcript, note_id="Note A")

    with console.status("[bold green]Generating Note B..."):
        note_b = gen_b.generate_note(transcript, note_id="Note B")

    # Display results
    rprint(
        Panel(
            note_a.raw_text,
            title=f"[bold green]Note A ({model_a})[/bold green]",
            border_style="green",
        )
    )
    rprint(
        Panel(
            note_b.raw_text, title=f"[bold blue]Note B ({model_b})[/bold blue]", border_style="blue"
        )
    )

    if output_file:
        data = {
            "transcript": transcript,
            "note_a": note_a.model_dump(),
            "note_b": note_b.model_dump(),
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        rprint(f"[bold green]Saved notes to {output_file}[/bold green]")


@app.command()
def judge(
    transcript_file: Optional[Path] = typer.Option(
        None, "--transcript-file", "-f", help="Path to dialogue transcript file."
    ),
    note_a_file: Optional[Path] = typer.Option(
        None, "--note-a", "-a", help="Path to Candidate Note A text file."
    ),
    note_b_file: Optional[Path] = typer.Option(
        None, "--note-b", "-b", help="Path to Candidate Note B text file."
    ),
    sample_index: int = typer.Option(
        0, "--sample-index", "-i", help="Index of sample fixture to arbitrate (0-2)."
    ),
    judge_model: str = typer.Option(
        "mock-judge", "--judge-model", "-m", help="LLM judge model identifier."
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional output JSON path for the arbitration verdict."
    ),
):
    """Run pairwise clinical arbitration with position-bias swap mitigation."""
    if transcript_file and note_a_file and note_b_file:
        transcript = transcript_file.read_text(encoding="utf-8")
        note_a = note_a_file.read_text(encoding="utf-8")
        note_b = note_b_file.read_text(encoding="utf-8")
        encounter_id = transcript_file.stem
    else:
        sample_path = (
            Path(__file__).resolve().parent.parent / "data" / "samples" / "aci_bench_samples.json"
        )
        if not sample_path.exists():
            rprint("[red]Error: Sample dataset not found.[/red]")
            raise typer.Exit(code=1)
        with open(sample_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        idx = min(sample_index, len(samples) - 1)
        sample = samples[idx]
        transcript = sample["transcript"]
        note_a = sample["note_a"]
        note_b = sample["note_b"]
        encounter_id = sample.get("encounter_id", f"Sample-{idx}")
        rprint(f"[yellow]Arbitrating bundled sample encounter: {encounter_id}[/yellow]")

    judge_instance = ClinicalJudge(model_name=judge_model)

    with console.status(
        f"[bold magenta]Arbitrating encounter {encounter_id} with dual-presentation swap mitigation..."
    ):
        verdict = judge_instance.arbitrate(
            transcript=transcript,
            note_a=note_a,
            note_b=note_b,
            encounter_id=encounter_id,
        )

    # Render summary table
    table = Table(title=f"Clinical Arbitration Results - {encounter_id}")
    table.add_column("Metric / Field", style="cyan", no_wrap=True)
    table.add_column("Value / Verdict", style="magenta")

    table.add_row(
        "Final Arbitrated Verdict", f"[bold green]{verdict.final_verdict.value}[/bold green]"
    )
    table.add_row("Winning Note", str(verdict.winning_note_id))
    table.add_row("Confidence Score", f"{verdict.confidence:.2%}")
    bias_color = "red" if verdict.position_bias_detected else "green"
    table.add_row(
        "Position Bias Detected", f"[{bias_color}]{verdict.position_bias_detected}[/{bias_color}]"
    )
    table.add_row("Presentation AB Winner", verdict.eval_ab.winner.value)
    table.add_row("Presentation BA Winner", verdict.eval_ba.winner.value)
    table.add_row("Note A Critical Errors", str(verdict.note_a_critical_count))
    table.add_row("Note B Critical Errors", str(verdict.note_b_critical_count))
    table.add_row(
        "Total Tokens", f"{verdict.total_prompt_tokens + verdict.total_completion_tokens:,}"
    )
    table.add_row("Total Inference Cost", f"${verdict.total_cost_usd:.5f}")

    console.print(table)
    rprint(
        Panel(
            verdict.clinical_rationale,
            title="[bold]Physician Clinical Argument[/bold]",
            border_style="cyan",
        )
    )

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(verdict.model_dump(), f, indent=2)
        rprint(f"[bold green]Saved verdict to {output_file}[/bold green]")


@app.command()
def validate_healthbench(
    samples_file: Optional[Path] = typer.Option(
        None, "--samples-file", "-f", help="Path to HealthBench samples JSON."
    ),
    judge_model: str = typer.Option(
        "mock-judge", "--judge-model", "-m", help="Model identifier for the evaluator."
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to write evaluation results."
    ),
):
    """Validate LLM judge against physician-written HealthBench gold standard rubrics."""
    if samples_file and samples_file.exists():
        path = samples_file
    else:
        path = (
            Path(__file__).resolve().parent.parent / "data" / "samples" / "healthbench_samples.json"
        )

    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    rprint(
        Panel(
            f"[bold green]Validating against {len(items)} HealthBench physician rubrics...[/bold green]"
        )
    )

    scorer = SingleNoteScorer(model_name=judge_model)

    physician_scores = []
    judge_scores = []
    physician_verdicts = []
    judge_verdicts = []

    for item in items:
        transcript = f"CLINICAL PROMPT / SCENARIO:\n{item['clinical_prompt']}\n\nPHYSICIAN GOLD CRITERIA:\n{item['physician_rubric']}"
        candidate_response = item["candidate_response"]

        eval_res = scorer.evaluate_note(
            transcript=transcript,
            note=candidate_response,
            note_id=item["item_id"],
        )

        gold_score = float(item["physician_gold_score"])
        j_score = eval_res.overall_score
        gold_v = "PASS" if gold_score >= 70.0 else "FAIL"
        j_v = "PASS" if j_score >= 70.0 else "FAIL"

        physician_scores.append(gold_score)
        judge_scores.append(j_score)
        physician_verdicts.append(gold_v)
        judge_verdicts.append(j_v)

    summary = generate_validation_summary(
        candidate_scores=judge_scores,
        gold_scores=physician_scores,
        candidate_verdicts=judge_verdicts,
        gold_verdicts=physician_verdicts,
    )

    # Render summary table
    table = Table(title="HealthBench Physician Validation Metrics")
    table.add_column("Statistical Metric", style="cyan")
    table.add_column("Measured Value", style="magenta")

    table.add_row("Evaluated Items", str(summary["sample_size"]))
    table.add_row("Pass/Fail Classification Accuracy", f"{summary['accuracy_pct']}%")
    table.add_row(
        "Cohen's Kappa (Inter-Rater Reliability)",
        f"{summary['cohens_kappa']} ({summary['kappa_interpretation']})",
    )
    table.add_row(
        "Spearman Rank Correlation (rho)",
        f"{summary['spearman_rho']} (p={summary['spearman_p_value']:.4f})",
    )

    console.print(table)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "summary": summary,
            "items": [
                {
                    "item_id": it["item_id"],
                    "physician_score": p_s,
                    "judge_score": j_s,
                    "physician_verdict": p_v,
                    "judge_verdict": j_v,
                    "agreement": p_v == j_v,
                }
                for it, p_s, j_s, p_v, j_v in zip(
                    items, physician_scores, judge_scores, physician_verdicts, judge_verdicts
                )
            ],
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        rprint(f"[bold green]Saved HealthBench validation results to {output_file}[/bold green]")


@app.command()
def report(
    evals_dir: Optional[Path] = typer.Option(
        None, "--evals-dir", "-d", help="Directory containing eval JSON files."
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output markdown report path."
    ),
):
    """Compile and display clinical benchmarking summary report."""
    target_dir = evals_dir or Path("evals")
    if not target_dir.exists():
        fallback_dir = Path(__file__).resolve().parent.parent / "evals"
        if fallback_dir.exists():
            target_dir = fallback_dir

    results_path = target_dir / "results.md"
    if results_path.exists():
        content = results_path.read_text(encoding="utf-8")
        console.print(content[:2500])
        rprint(f"[green]Full report available at: {results_path}[/green]")
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            rprint(f"[bold green]Report exported to {output_file}[/bold green]")
    else:
        rprint(f"[yellow]No results.md found in {target_dir}. Run evaluations first.[/yellow]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn hot reloading."),
):
    """Launch the interactive Clinical Note Arbitration FastAPI server and web dashboard."""
    import uvicorn

    rprint(
        Panel(
            f"[bold green]Starting Clinical Note Arbitration Server at http://{host}:{port}[/bold green]\nInteractive dashboard: http://{host}:{port}/dashboard\nAPI Documentation: http://{host}:{port}/docs"
        )
    )
    uvicorn.run("arbiter.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
