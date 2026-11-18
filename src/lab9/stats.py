"""Statistics students implement (M2 judge calibration, M4 paired bootstrap).

Rules: a judge is a classifier -- measure it like one. A higher mean is not
"better" until the paired CI of the difference excludes 0.
Reference values to self-check are in tests/test_lab9_stats.py.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import numbers

POS = "FAIL"   # the judge's job is to FIND failures, so FAIL is the positive class


def _validate_labels(human: Sequence[str], judge: Sequence[str]) -> None:
    if len(human) != len(judge):
        raise ValueError("human and judge must contain the same number of labels")
    valid = {"FAIL", "PASS"}
    if any(label not in valid for label in (*human, *judge)):
        raise ValueError("labels must be 'FAIL' or 'PASS'")


def confusion(human: Sequence[str], judge: Sequence[str]) -> dict[str, int]:
    """tp/fp/fn/tn with FAIL as positive."""
    _validate_labels(human, judge)
    tp = fp = fn = tn = 0
    for actual, predicted in zip(human, judge):
        if actual == POS and predicted == POS:
            tp += 1
        elif actual != POS and predicted == POS:
            fp += 1
        elif actual == POS and predicted != POS:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall(human: Sequence[str], judge: Sequence[str]) -> tuple[float, float]:
    cm = confusion(human, judge)
    precision_denominator = cm["tp"] + cm["fp"]
    recall_denominator = cm["tp"] + cm["fn"]
    precision = cm["tp"] / precision_denominator if precision_denominator else 0.0
    recall = cm["tp"] / recall_denominator if recall_denominator else 0.0
    return precision, recall


def cohen_kappa(human: Sequence[str], judge: Sequence[str]) -> float:
    """kappa = (p_o - p_e) / (1 - p_e). Raise ValueError when undefined (p_e == 1):
    a constant judge has no signal and must FAIL calibration, not pass on a technicality."""
    cm = confusion(human, judge)
    n = len(human)
    if n == 0:
        raise ValueError("kappa is undefined for an empty sample")
    observed = (cm["tp"] + cm["tn"]) / n
    human_fail = (cm["tp"] + cm["fn"]) / n
    judge_fail = (cm["tp"] + cm["fp"]) / n
    expected = human_fail * judge_fail + (1 - human_fail) * (1 - judge_fail)
    if expected == 1:
        raise ValueError("kappa is undefined when expected agreement is 1")
    return (observed - expected) / (1 - expected)


def rogan_gladen(observed_rate: float, sensitivity: float, specificity: float) -> float:
    """Correct an observed failure rate for judge error; clip to [0, 1]."""
    rates = (observed_rate, sensitivity, specificity)
    if any(not math.isfinite(rate) or not 0 <= rate <= 1 for rate in rates):
        raise ValueError("observed_rate, sensitivity, and specificity must be in [0, 1]")
    denominator = sensitivity + specificity - 1
    if denominator == 0:
        raise ValueError("failure rate is not identifiable when sensitivity + specificity is 1")
    corrected = (observed_rate + specificity - 1) / denominator
    return min(1.0, max(0.0, corrected))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 1000,
                     rng: int = 2027, confidence: float = 0.95) -> tuple[float, float, float]:
    """a[i], b[i] = per-item mean over seeds, SAME items, SAME order.
    Return (mean(a)-mean(b), lo, hi) with scipy.stats.bootstrap(paired=True, method="percentile",
    rng=numpy.random.default_rng(rng)). If a == b everywhere return (0.0, 0.0, 0.0)."""
    import numpy as np
    from scipy.stats import bootstrap

    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    if len(a) == 0:
        raise ValueError("paired samples must not be empty")
    if not isinstance(n_resamples, numbers.Integral) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired samples must contain only finite values")
    difference = float(np.mean(left) - np.mean(right))
    if np.array_equal(left, right):
        return 0.0, 0.0, 0.0

    def mean_difference(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    result = bootstrap(
        (left, right),
        mean_difference,
        n_resamples=int(n_resamples),
        confidence_level=confidence,
        method="percentile",
        paired=True,
        vectorized=True,
        rng=np.random.default_rng(rng),
    )
    return difference, float(result.confidence_interval.low), float(result.confidence_interval.high)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that AT LEAST ONE of k trials drawn from n (c of them passing) passes.
    Unbiased estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k). Raise ValueError if k > n."""
    _validate_pass_k_args(n, c, k)
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k (tau-bench, Yao et al. 2024): probability that ALL k trials drawn from n pass.
    Unbiased estimator: C(c, k) / C(n, k). Raise ValueError if k > n.
    pass^k <= pass@1 <= pass@k; a reliable agent keeps pass^k close to pass@1."""
    _validate_pass_k_args(n, c, k)
    return math.comb(c, k) / math.comb(n, k)


def _validate_pass_k_args(n: int, c: int, k: int) -> None:
    if any(not isinstance(value, numbers.Integral) or isinstance(value, bool) for value in (n, c, k)):
        raise ValueError("n, c, and k must be integers")
    if n <= 0 or c < 0 or c > n or k <= 0 or k > n:
        raise ValueError("require n > 0, 0 <= c <= n, and 1 <= k <= n")
