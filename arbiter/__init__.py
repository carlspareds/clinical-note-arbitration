"""Clinical Note Arbitration package: LLM-as-a-Judge for clinical notes."""

from arbiter.generator import NoteGenerator
from arbiter.judge import ClinicalJudge
from arbiter.models import CandidateNote, PairwiseVerdict, SingleNoteEvaluation
from arbiter.scorer import SingleNoteScorer
from arbiter.stats import (
    calculate_accuracy,
    calculate_cohens_kappa,
    calculate_position_bias_rate,
    calculate_spearman_rank_correlation,
)

__version__ = "0.1.0"

__all__ = [
    "ClinicalJudge",
    "SingleNoteScorer",
    "NoteGenerator",
    "CandidateNote",
    "PairwiseVerdict",
    "SingleNoteEvaluation",
    "calculate_accuracy",
    "calculate_cohens_kappa",
    "calculate_spearman_rank_correlation",
    "calculate_position_bias_rate",
]
