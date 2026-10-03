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


def calculate_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a binomial proportion.
    Returns: (lower_bound_pct, upper_bound_pct) in percentage [0.0, 100.0].
    """
    if n <= 0:
        return 0.0, 0.0

    k = max(0, min(k, n))
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = k / n
    denominator = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denominator
    half_width = (z * np.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denominator

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return round(float(lower) * 100.0, 2), round(float(upper) * 100.0, 2)


def calculate_accuracy_ci(
    predictions: Sequence[Any], ground_truth: Sequence[Any], confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate 95% Wilson confidence interval for classification accuracy percentage."""
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0, 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return calculate_wilson_ci(correct, len(predictions), confidence=confidence)


def calculate_cohens_kappa_ci(
    rater1: Sequence[Any],
    rater2: Sequence[Any],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Calculate non-parametric bootstrap confidence interval for Cohen's Kappa.
    Returns: (lower_bound, upper_bound) rounded to 4 decimals.
    """
    if len(rater1) != len(rater2) or len(rater1) <= 1:
        kappa = calculate_cohens_kappa(rater1, rater2)
        return round(kappa, 4), round(kappa, 4)

    n = len(rater1)
    r1_arr = np.array(rater1)
    r2_arr = np.array(rater2)

    rng = np.random.RandomState(seed)
    boot_kappas = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        b_k = calculate_cohens_kappa(r1_arr[idx], r2_arr[idx])
        boot_kappas.append(b_k)

    alpha = 1.0 - confidence
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    lower = float(np.percentile(boot_kappas, lower_pct))
    upper = float(np.percentile(boot_kappas, upper_pct))

    lower = max(-1.0, min(1.0, lower))
    upper = max(-1.0, min(1.0, upper))

    if lower > upper:
        lower, upper = upper, lower

    return round(lower, 4), round(upper, 4)


def calculate_position_bias_rate(swap_results: Sequence[Dict[str, Any]]) -> float:
    """
    Calculate the percentage of encounters where changing the order (AB vs BA) caused an inconsistent outcome.
    swap_results items should contain {'position_bias_detected': bool}
    """
    if not swap_results:
        return 0.0
    bias_count = sum(1 for r in swap_results if r.get("position_bias_detected", False))
    return round((bias_count / len(swap_results)) * 100.0, 2)


def calculate_position_bias_ci(
    swap_results: Sequence[Dict[str, Any]], confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate 95% Wilson confidence interval for position bias rate percentage."""
    if not swap_results:
        return 0.0, 0.0
    bias_count = sum(1 for r in swap_results if r.get("position_bias_detected", False))
    return calculate_wilson_ci(bias_count, len(swap_results), confidence=confidence)


def generate_validation_summary(
    candidate_scores: Sequence[float],
    gold_scores: Sequence[float],
    candidate_verdicts: Sequence[Any],
    gold_verdicts: Sequence[Any],
    swap_results: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate comprehensive statistical validation report."""
    acc = calculate_accuracy(candidate_verdicts, gold_verdicts)
    acc_ci = calculate_accuracy_ci(candidate_verdicts, gold_verdicts)
    kappa = calculate_cohens_kappa(candidate_verdicts, gold_verdicts)
    kappa_ci = calculate_cohens_kappa_ci(candidate_verdicts, gold_verdicts)
    rho, p_val = calculate_spearman_rank_correlation(candidate_scores, gold_scores)
    bias_rate = calculate_position_bias_rate(swap_results) if swap_results else 0.0
    bias_ci = calculate_position_bias_ci(swap_results) if swap_results else (0.0, 0.0)

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
        "accuracy_ci_95": list(acc_ci),
        "cohens_kappa": kappa,
        "cohens_kappa_ci_95": list(kappa_ci),
        "kappa_interpretation": kappa_interp,
        "spearman_rho": rho,
        "spearman_p_value": p_val,
        "position_bias_rate_pct": bias_rate,
        "position_bias_ci_95": list(bias_ci) if swap_results else [0.0, 0.0],
    }
