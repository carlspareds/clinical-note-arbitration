"""Unit tests for statistical inter-rater agreement, accuracy, and position bias metrics."""

from arbiter.stats import (
    calculate_accuracy,
    calculate_cohens_kappa,
    calculate_position_bias_rate,
    calculate_spearman_rank_correlation,
    generate_validation_summary,
)


def test_calculate_accuracy():
    preds = ["PASS", "FAIL", "PASS", "PASS"]
    truth = ["PASS", "FAIL", "PASS", "FAIL"]
    acc = calculate_accuracy(preds, truth)
    assert acc == 75.0


def test_calculate_cohens_kappa():
    # Perfect agreement
    r1 = ["PASS", "FAIL", "PASS", "PASS"]
    r2 = ["PASS", "FAIL", "PASS", "PASS"]
    assert calculate_cohens_kappa(r1, r2) == 1.0

    # Partial agreement
    r1 = ["PASS", "PASS", "FAIL", "FAIL"]
    r2 = ["PASS", "FAIL", "FAIL", "PASS"]
    kappa = calculate_cohens_kappa(r1, r2)
    assert -1.0 <= kappa <= 1.0


def test_calculate_spearman_rank_correlation():
    scores1 = [10.0, 20.0, 30.0, 40.0, 50.0]
    scores2 = [12.0, 22.0, 31.0, 44.0, 55.0]
    rho, p_val = calculate_spearman_rank_correlation(scores1, scores2)
    assert rho == 1.0
    assert p_val < 0.05


def test_calculate_position_bias_rate():
    swaps = [
        {"position_bias_detected": False},
        {"position_bias_detected": True},
        {"position_bias_detected": False},
        {"position_bias_detected": False},
    ]
    rate = calculate_position_bias_rate(swaps)
    assert rate == 25.0


def test_generate_validation_summary():
    c_scores = [90.0, 80.0, 20.0]
    g_scores = [95.0, 85.0, 15.0]
    c_verdicts = ["PASS", "PASS", "FAIL"]
    g_verdicts = ["PASS", "PASS", "FAIL"]
    summary = generate_validation_summary(c_scores, g_scores, c_verdicts, g_verdicts)
    assert summary["sample_size"] == 3
    assert summary["accuracy_pct"] == 100.0
    assert summary["cohens_kappa"] == 1.0


def test_stats_edge_cases():
    # Empty inputs
    assert calculate_accuracy([], []) == 0.0
    assert calculate_cohens_kappa([], []) == 0.0
    assert calculate_position_bias_rate([]) == 0.0

    rho, p_val = calculate_spearman_rank_correlation([], [])
    assert rho == 0.0
    assert p_val == 1.0

    # Zero variance (all scores identical)
    rho_const, p_const = calculate_spearman_rank_correlation([50.0, 50.0, 50.0], [50.0, 50.0, 50.0])
    assert rho_const == 1.0
    assert p_const == 0.0

    # Single category kappa
    assert calculate_cohens_kappa(["PASS", "PASS"], ["PASS", "PASS"]) == 1.0
