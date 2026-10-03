"""Unit tests for pairwise clinical judge with position-bias mitigation."""

from arbiter.judge import ClinicalJudge
from arbiter.models import CandidateNote, FinalVerdictEnum, PairwiseVerdict


def test_clinical_judge_arbitration(
    sample_transcript: str,
    safe_note: CandidateNote,
    dangerous_note: CandidateNote,
):
    judge = ClinicalJudge(model_name="mock-judge")
    verdict = judge.arbitrate(
        transcript=sample_transcript,
        note_a=safe_note,
        note_b=dangerous_note,
        encounter_id="TEST-001",
    )

    assert isinstance(verdict, PairwiseVerdict)
    assert verdict.encounter_id == "TEST-001"
    assert verdict.eval_ab is not None
    assert verdict.eval_ba is not None

    # Safe note should win over dangerous note
    assert verdict.final_verdict == FinalVerdictEnum.NOTE_A_WINS
    assert verdict.winning_note_id == "Note A"
    assert verdict.confidence > 0.70
    assert not verdict.position_bias_detected
    assert len(verdict.clinical_rationale) > 20


def test_pairwise_verdict_serialization(
    sample_transcript: str,
    safe_note: CandidateNote,
    dangerous_note: CandidateNote,
):
    judge = ClinicalJudge(model_name="mock-judge")
    verdict = judge.arbitrate(sample_transcript, safe_note, dangerous_note)
    dumped = verdict.model_dump()
    assert "final_verdict" in dumped
    assert "eval_ab" in dumped
    assert "eval_ba" in dumped
    assert "total_cost_usd" in dumped


def test_clinical_judge_reversed_notes(
    sample_transcript: str,
    safe_note: CandidateNote,
    dangerous_note: CandidateNote,
):
    """When Note A is dangerous and Note B is safe, Note B must win."""
    judge = ClinicalJudge(model_name="mock-judge")
    verdict = judge.arbitrate(
        transcript=sample_transcript,
        note_a=dangerous_note,
        note_b=safe_note,
        encounter_id="TEST-REVERSED",
    )
    assert verdict.final_verdict == FinalVerdictEnum.NOTE_B_WINS
    assert verdict.winning_note_id == "Note B"
    assert verdict.note_a_critical_count >= 1
    assert verdict.note_b_critical_count == 0


def test_position_bias_detection_invariant(
    sample_transcript: str,
    safe_note: CandidateNote,
):
    """When two identical safe notes are arbitrated, verdict is a tie or consistent without bias."""
    judge = ClinicalJudge(model_name="mock-judge")
    verdict = judge.arbitrate(
        transcript=sample_transcript,
        note_a=safe_note,
        note_b=safe_note,
        encounter_id="TEST-IDENTICAL",
    )
    assert verdict.final_verdict in [
        FinalVerdictEnum.TIE,
        FinalVerdictEnum.NOTE_A_WINS,
        FinalVerdictEnum.NOTE_B_WINS,
    ]
    assert verdict.confidence >= 0.50


def test_position_bias_flip_triggers_inconclusive(sample_transcript: str):
    """When a model consistently selects Option 1 (first-position bias), the arbiter detects
    the order reversal and returns INCONCLUSIVE_POSITION_BIAS with no winner."""
    from arbiter.llm.providers import BaseLLMProvider, LLMResponse

    class FirstOptionBiasedProvider(BaseLLMProvider):
        def generate(self, prompt: str, **kwargs) -> LLMResponse:
            content = '{"winner": "Option 1", "margin": 25.0, "clinical_rationale": "Always favors Option 1 regardless of content."}'
            return LLMResponse(
                content=content,
                parsed_json={
                    "winner": "Option 1",
                    "margin": 25.0,
                    "clinical_rationale": "Always favors Option 1 regardless of content.",
                },
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.0,
                latency_seconds=0.01,
                model="biased-model",
                provider="mock",
            )

    judge = ClinicalJudge(model_name="mock-judge")
    judge.client.provider = FirstOptionBiasedProvider(model_name="biased-model")

    verdict = judge.arbitrate(
        transcript=sample_transcript,
        note_a="SOAP Note A content",
        note_b="SOAP Note B content",
        encounter_id="BIAS-INCONCL-001",
    )

    assert verdict.position_bias_detected is True
    assert verdict.final_verdict == FinalVerdictEnum.INCONCLUSIVE_POSITION_BIAS
    assert verdict.winning_note_id is None
    assert verdict.winning_model is None
    assert verdict.confidence <= 0.50
    assert "Position bias detected" in verdict.clinical_rationale
    assert "Presentation AB selected 'A'" in verdict.clinical_rationale
    assert "Presentation BA selected 'B'" in verdict.clinical_rationale

