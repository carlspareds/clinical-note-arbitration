"""Shared pytest fixtures for clinical note arbitration tests."""

import pytest

from arbiter.llm.client import UnifiedLLMClient, get_llm_client
from arbiter.models import CandidateNote, SOAPSections
from rubric.rubric_loader import ClinicalRubric, load_clinical_rubric


@pytest.fixture
def mock_client() -> UnifiedLLMClient:
    """Return an offline deterministic mock LLM client."""
    return get_llm_client("mock-judge")


@pytest.fixture
def clinical_rubric() -> ClinicalRubric:
    """Load default physician clinical rubric."""
    return load_clinical_rubric()


@pytest.fixture
def sample_transcript() -> str:
    return (
        "Doctor: Hello Mr. Smith, what brings you in?\n"
        "Patient: I've had a bad dry cough for 3 weeks and some shortness of breath.\n"
        "Doctor: Are you allergic to anything?\n"
        "Patient: Yes, I am severely allergic to Penicillin. I had hives and swelling.\n"
        "Doctor: What medications do you take?\n"
        "Patient: Lisinopril 20mg daily.\n"
        "Doctor: We will stop Lisinopril and start Losartan 50mg. Go to the ED if chest pain develops."
    )


@pytest.fixture
def safe_note() -> CandidateNote:
    raw = (
        "SUBJECTIVE:\n"
        "Patient presents with 3-week dry cough on Lisinopril 20mg.\n"
        "Allergies: Penicillin (severe anaphylactoid reaction, hives).\n\n"
        "OBJECTIVE:\n"
        "Vitals stable, lungs clear.\n\n"
        "ASSESSMENT:\n"
        "Lisinopril-induced cough. Severe Penicillin allergy.\n\n"
        "PLAN:\n"
        "Discontinue Lisinopril. Start Losartan 50mg daily. Strict ED precautions."
    )
    return CandidateNote(
        note_id="Note Safe",
        model_name="safe-model",
        raw_text=raw,
        sections=SOAPSections(
            subjective="Patient presents with 3-week dry cough on Lisinopril 20mg. Allergies: Penicillin.",
            objective="Vitals stable, lungs clear.",
            assessment="Lisinopril-induced cough.",
            plan="Discontinue Lisinopril. Start Losartan 50mg daily.",
        ),
    )


@pytest.fixture
def dangerous_note() -> CandidateNote:
    raw = (
        "SUBJECTIVE:\n"
        "Patient presents with cough.\n"
        "Allergies: NKDA (No Known Drug Allergies).\n\n"
        "OBJECTIVE:\n"
        "Lungs clear.\n\n"
        "ASSESSMENT:\n"
        "Bacterial bronchitis.\n\n"
        "PLAN:\n"
        "Prescribe Amoxicillin 500mg TID. Continue Lisinopril."
    )
    return CandidateNote(
        note_id="Note Dangerous",
        model_name="dangerous-model",
        raw_text=raw,
        sections=SOAPSections(
            subjective="Patient presents with cough. Allergies: NKDA.",
            objective="Lungs clear.",
            assessment="Bacterial bronchitis.",
            plan="Prescribe Amoxicillin 500mg TID.",
        ),
    )
