"""Unit tests for clinical rubric parsing and schema validation."""

from rubric.rubric_loader import ClinicalRubric, load_clinical_rubric


def test_load_clinical_rubric():
    rubric = load_clinical_rubric()
    assert isinstance(rubric, ClinicalRubric)
    assert len(rubric.criteria) >= 6
    assert "CRITICAL" in rubric.severity_levels
    assert "MAJOR" in rubric.severity_levels
    assert "MINOR" in rubric.severity_levels

    # Check criteria sum
    weights = [c.weight for c in rubric.criteria]
    assert sum(weights) == 100

    # Check allergy criterion
    allergy = rubric.get_criterion("ALLERGIES_AND_INTOLERANCES")
    assert allergy is not None
    assert allergy.weight >= 20


def test_rubric_prompt_text(clinical_rubric: ClinicalRubric):
    prompt_text = clinical_rubric.get_prompt_text()
    assert "CRITICAL" in prompt_text
    assert "ALLERGIES_AND_INTOLERANCES" in prompt_text
    assert "MEDICATION_ACCURACY" in prompt_text
