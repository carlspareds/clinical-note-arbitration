"""Pydantic schemas for clinical notes, safety violations, evaluations, and arbitration verdicts."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SeverityEnum(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    BENIGN = "BENIGN"


class CriterionIdEnum(str, Enum):
    ALLERGIES_AND_INTOLERANCES = "ALLERGIES_AND_INTOLERANCES"
    MEDICATION_ACCURACY = "MEDICATION_ACCURACY"
    RED_FLAGS_AND_ACUTE_RISKS = "RED_FLAGS_AND_ACUTE_RISKS"
    PLAN_SAFETY_AND_CONTRAINDICATIONS = "PLAN_SAFETY_AND_CONTRAINDICATIONS"
    HALLUCINATED_FINDINGS = "HALLUCINATED_FINDINGS"
    OMITTED_FINDINGS_AND_COMPLETENESS = "OMITTED_FINDINGS_AND_COMPLETENESS"
    INTERNAL_CONSISTENCY_AND_STRUCTURE = "INTERNAL_CONSISTENCY_AND_STRUCTURE"


class SafetyViolation(BaseModel):
    criterion_id: str = Field(..., description="ID of violated clinical criterion")
    severity: SeverityEnum = Field(..., description="Severity classification")
    description: str = Field(..., description="Clinical description of the error")
    excerpt_quote: Optional[str] = Field(None, description="Exact phrase from note or transcript")
    clinical_consequence: str = Field(..., description="Potential clinical harm or patient risk")


class SOAPSections(BaseModel):
    subjective: str = Field(..., description="Subjective history and chief complaint")
    objective: str = Field(..., description="Objective physical exam and vitals")
    assessment: str = Field(..., description="Assessment and diagnostic impression")
    plan: str = Field(..., description="Management plan, prescriptions, and follow-up")


class CandidateNote(BaseModel):
    note_id: str = Field(..., description="Identifier for the candidate note (e.g. Note A)")
    model_name: str = Field(..., description="Model that generated this note")
    raw_text: str = Field(..., description="Full text of the SOAP note")
    sections: Optional[SOAPSections] = Field(None, description="Parsed SOAP sections if extracted")


class CriterionScore(BaseModel):
    criterion_id: str
    criterion_name: str
    weight: int
    score: float = Field(..., ge=0.0, le=100.0, description="Raw score between 0 and 100")
    weighted_score: float = Field(..., ge=0.0, description="Score multiplied by normalized weight")
    violations: List[SafetyViolation] = Field(default_factory=list)
    clinical_rationale: str = Field(
        ..., description="Physician justification for the criterion score"
    )


class SingleNoteEvaluation(BaseModel):
    note_id: str
    model_name: str
    overall_score: float = Field(..., ge=0.0, le=100.0)
    has_critical_error: bool = Field(default=False)
    critical_error_count: int = Field(default=0)
    major_error_count: int = Field(default=0)
    minor_error_count: int = Field(default=0)
    criterion_scores: List[CriterionScore] = Field(default_factory=list)
    all_violations: List[SafetyViolation] = Field(default_factory=list)
    clinical_summary: str = Field(..., description="Executive clinical evaluation summary")
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    cost_usd: float = Field(default=0.0)


class OrderVerdict(str, Enum):
    A = "A"
    B = "B"
    TIE = "TIE"


class OrderEvaluation(BaseModel):
    order: str = Field(..., description="Ordering presentation: 'AB' or 'BA'")
    winner: OrderVerdict = Field(
        ..., description="Winner in this presentation ('A', 'B', or 'TIE')"
    )
    margin: float = Field(default=0.0, description="Confidence / quality margin between notes")
    rationale: str = Field(..., description="Physician-level judge reasoning for this ordering")
    note_a_violations: List[SafetyViolation] = Field(default_factory=list)
    note_b_violations: List[SafetyViolation] = Field(default_factory=list)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    cost_usd: float = Field(default=0.0)


class FinalVerdictEnum(str, Enum):
    NOTE_A_WINS = "NOTE_A_WINS"
    NOTE_B_WINS = "NOTE_B_WINS"
    TIE = "TIE"
    INCONCLUSIVE_POSITION_BIAS = "INCONCLUSIVE_POSITION_BIAS"


class PairwiseVerdict(BaseModel):
    encounter_id: Optional[str] = Field(None, description="Encounter / transcript identifier")
    final_verdict: FinalVerdictEnum = Field(..., description="Final arbitrated verdict")
    winning_note_id: Optional[str] = Field(
        None, description="'Note A', 'Note B', or None if tie/bias"
    )
    winning_model: Optional[str] = Field(None, description="Model of the winning note")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    position_bias_detected: bool = Field(
        ..., description="True if swap presentation resulted in contradiction"
    )
    eval_ab: OrderEvaluation = Field(
        ..., description="Results when Note A presented as Option 1, Note B as Option 2"
    )
    eval_ba: OrderEvaluation = Field(
        ..., description="Results when Note B presented as Option 1, Note A as Option 2"
    )
    clinical_rationale: str = Field(..., description="Comprehensive clinical safety argument")
    safety_comparison_summary: str = Field(
        ..., description="Direct comparison of safety violations and omission risks"
    )
    note_a_critical_count: int = Field(default=0)
    note_b_critical_count: int = Field(default=0)
    total_prompt_tokens: int = Field(default=0)
    total_completion_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)
    judge_model: str = Field(..., description="Model identifier of the LLM judge")


class HealthBenchItemValidation(BaseModel):
    item_id: str
    domain: str
    physician_score: float
    judge_score: float
    agreement: bool
    judge_verdict: str
    physician_rubric: str
    clinical_notes: str
