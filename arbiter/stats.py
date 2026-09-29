"""Statistical validation metrics: Cohen's kappa, Spearman rank correlation, accuracy, and position bias rate."""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def calculate_accuracy(predictions: Sequence[Any], ground_truth: Sequence[Any]) -> float:
    """Compute basic classification accuracy percentage between 0.0 and 100.0."""
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return round((correct / len(predictions)) * 100.0, 2)


def calculate_cohens_kappa(rater1: Sequence[Any], rater2: Sequence[Any]) -> float:
    """
    Calculate Cohen's Kappa coefficient for inter-rater agreement.
    kappa = (P_o - P_e) / (1 - P_e)
    Range: -1.0 to +1.0 (>0.80 = almost perfect agreement).
    """
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return 0.0

    categories = sorted(list(set(rater1).union(set(rater2))))
    n = len(rater1)
    if n <= 1 or len(categories) <= 1:
        return 1.0 if list(rater1) == list(rater2) else 0.0

    cat_idx = {cat: idx for idx, cat in enumerate(categories)}
    num_cats = len(categories)

    # Build confusion matrix
    matrix = np.zeros((num_cats, num_cats), dtype=int)
    for r1, r2 in zip(rater1, rater2):
        matrix[cat_idx[r1], cat_idx[r2]] += 1

    # Observed agreement
    p_o = np.trace(matrix) / n

    # Expected agreement
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    p_e = np.sum((row_sums * col_sums) / (n * n))

    if abs(1.0 - p_e) < 1e-9:
        return 1.0 if p_o == 1.0 else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    if np.isnan(kappa):
        return 0.0
    return round(float(kappa), 4)


def calculate_spearman_rank_correlation(
    scores1: Sequence[float], scores2: Sequence[float]
) -> Tuple[float, float]:
    """
    Calculate Spearman's rank correlation coefficient (rho) and p-value between two sets of continuous grades.
    Returns: (rho, p_value)
    """
    if len(scores1) != len(scores2) or len(scores1) < 2:
        return 0.0, 1.0

    arr1 = np.array(scores1, dtype=float)
    arr2 = np.array(scores2, dtype=float)

    # Check for zero variance
    if np.std(arr1) == 0.0 or np.std(arr2) == 0.0:
        return (1.0, 0.0) if np.all(arr1 == arr2) else (0.0, 1.0)

    res = stats.spearmanr(arr1, arr2)
    rho = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
    p_val = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])

    if np.isnan(rho):
        rho = 0.0
    if np.isnan(p_val):
        p_val = 1.0

    return round(rho, 4), round(p_val, 6)


def calculate_position_bias_rate(swap_results: Sequence[Dict[str, Any]]) -> float:
    """
    Calculate the percentage of encounters where changing the order (AB vs BA) caused an inconsistent outcome.
    swap_results items should contain {'position_bias_detected': bool}
    """
    if not swap_results:
        return 0.0
    bias_count = sum(1 for r in swap_results if r.get("position_bias_detected", False))
    return round((bias_count / len(swap_results)) * 100.0, 2)


def generate_validation_summary(
    candidate_scores: Sequence[float],
    gold_scores: Sequence[float],
    candidate_verdicts: Sequence[Any],
    gold_verdicts: Sequence[Any],
    swap_results: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate comprehensive statistical validation report."""
    acc = calculate_accuracy(candidate_verdicts, gold_verdicts)
    kappa = calculate_cohens_kappa(candidate_verdicts, gold_verdicts)
    rho, p_val = calculate_spearman_rank_correlation(candidate_scores, gold_scores)
    bias_rate = calculate_position_bias_rate(swap_results) if swap_results else 0.0

    # Interpretation
    if kappa >= 0.81:
        kappa_interp = "Almost Perfect Agreement"
    elif kappa >= 0.61:
        kappa_interp = "Substantial Agreement"
    elif kappa >= 0.41:
        kappa_interp = "Moderate Agreement"
    else:
        kappa_interp = "Fair or Poor Agreement"

    return {
        "sample_size": len(candidate_scores),
        "accuracy_pct": acc,
        "cohens_kappa": kappa,
        "kappa_interpretation": kappa_interp,
        "spearman_rho": rho,
        "spearman_p_value": p_val,
        "position_bias_rate_pct": bias_rate,
    }
