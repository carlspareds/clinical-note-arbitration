"""Unit tests for single-note rubric evaluation and clinical safety scoring."""

from arbiter.models import CandidateNote, SingleNoteEvaluation
from arbiter.scorer import SingleNoteScorer


def test_scorer_safe_note(sample_transcript: str, safe_note: CandidateNote):
    scorer = SingleNoteScorer(model_name="mock-judge")
    eval_res = scorer.evaluate_note(
        transcript=sample_transcript,
        note=safe_note,
        note_id="Safe Note",
    )
    assert isinstance(eval_res, SingleNoteEvaluation)
    assert eval_res.overall_score >= 80.0
    assert not eval_res.has_critical_error
    assert len(eval_res.criterion_scores) > 0


def test_scorer_dangerous_note(sample_transcript: str, dangerous_note: CandidateNote):
    scorer = SingleNoteScorer(model_name="mock-judge")
    eval_res = scorer.evaluate_note(
        transcript=sample_transcript,
        note=dangerous_note,
        note_id="Dangerous Note",
    )
    assert isinstance(eval_res, SingleNoteEvaluation)
    # Dangerous note has missed allergy and penicillin prescription
    assert eval_res.has_critical_error
    assert eval_res.overall_score < 75.0
    assert eval_res.critical_error_count >= 1


def test_scorer_string_input(sample_transcript: str):
    """Verify that SingleNoteScorer accepts raw string notes."""
    scorer = SingleNoteScorer(model_name="mock-judge")
    eval_res = scorer.evaluate_note(
        transcript=sample_transcript,
        note="SUBJECTIVE: Cough on Lisinopril. Allergies: Penicillin.\nPLAN: Switch to Losartan.",
        note_id="String Note",
    )
    assert isinstance(eval_res, SingleNoteEvaluation)
    assert eval_res.overall_score >= 80.0
    assert not eval_res.has_critical_error
