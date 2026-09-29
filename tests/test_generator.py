"""Unit tests for clinical SOAP note generation and section parsing."""

from arbiter.generator import NoteGenerator, parse_soap_sections
from arbiter.models import CandidateNote


def test_parse_soap_sections():
    sample_text = """
    SUBJECTIVE:
    Patient reports 3 days of sore throat and fever.
    OBJECTIVE:
    T 38.2 C, pharyngeal erythema with exudate.
    ASSESSMENT:
    Acute streptococcal pharyngitis.
    PLAN:
    Rapid strep swab, symptomatic relief.
    """
    sections = parse_soap_sections(sample_text)
    assert sections is not None
    assert "sore throat" in sections.subjective
    assert "pharyngeal erythema" in sections.objective
    assert "streptococcal" in sections.assessment
    assert "Rapid strep" in sections.plan


def test_note_generator(sample_transcript: str):
    generator = NoteGenerator(model_name="mock-generator")
    note = generator.generate_note(sample_transcript, note_id="Test Note")
    assert isinstance(note, CandidateNote)
    assert note.note_id == "Test Note"
    assert len(note.raw_text) > 50
    assert note.sections is not None
