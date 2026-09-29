"""Unit tests for the Typer CLI application."""

from typer.testing import CliRunner

from arbiter.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Clinical Note Arbitration CLI" in result.stdout
    assert "generate" in result.stdout
    assert "judge" in result.stdout
    assert "validate-healthbench" in result.stdout
    assert "report" in result.stdout


def test_cli_generate():
    result = runner.invoke(
        app,
        [
            "generate",
            "--text",
            "Doctor: Hello. Patient: I have a cough.",
            "--model-a",
            "mock-generator",
            "--model-b",
            "mock-generator",
        ],
    )
    assert result.exit_code == 0
    assert "Clinical Note Generation" in result.stdout
    assert "Note A" in result.stdout
    assert "Note B" in result.stdout


def test_cli_judge_sample():
    result = runner.invoke(
        app,
        ["judge", "--sample-index", "0", "--judge-model", "mock-judge"],
    )
    assert result.exit_code == 0
    assert "Clinical Arbitration Results" in result.stdout
    assert "Final Arbitrated Verdict" in result.stdout
    assert "NOTE_A_WINS" in result.stdout


def test_cli_validate_healthbench():
    result = runner.invoke(
        app,
        ["validate-healthbench", "--judge-model", "mock-judge"],
    )
    assert result.exit_code == 0
    assert "HealthBench Physician Validation Metrics" in result.stdout
    assert "Cohen's Kappa" in result.stdout


def test_cli_report(tmp_path):
    out_file = tmp_path / "test_report.md"
    result = runner.invoke(
        app,
        ["report", "--output", str(out_file)],
    )
    assert result.exit_code == 0
    assert "Clinical AI Evaluation" in result.stdout
    assert out_file.exists()
    assert len(out_file.read_text(encoding="utf-8")) > 100
