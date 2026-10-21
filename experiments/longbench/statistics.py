"""Summary statistics across repeated runs."""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence


def _moments(values: Sequence[float]) -> dict:
    n = len(values)
    sample_sd = statistics.stdev(values) if n > 1 else None
    return {
        "n": n,
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "variance_population": statistics.pvariance(values),
        "sd_population": statistics.pstdev(values),
        "variance_sample": statistics.variance(values) if n > 1 else None,
        "sd_sample": sample_sd,
        "sem": sample_sd / math.sqrt(n) if sample_sd is not None else None,
        "unit": "fraction",
    }


def summarize_repeats(scores: Mapping[str, float], expected: Sequence[str]) -> dict:
    """Summarize scores from complete suite runs under the same configuration."""
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected repeat labels must be nonempty and unique")
    if set(scores) != set(expected):
        raise ValueError("missing or unexpected repeat labels")
    values = []
    for label in expected:
        value = scores[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("repeat scores must be numeric rates")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("repeat scores must be finite rates in [0, 1]")
        values.append(float(value))
    return {
        **_moments(values),
        "repeat_labels": list(expected),
        "repeat_scores": values,
    }


def paired_effect(
    baseline: Mapping[str, float], candidate: Mapping[str, float], expected: Sequence[str]
) -> dict:
    """Compare candidate and baseline scores on matched seeds."""
    left = summarize_repeats(baseline, expected)
    right = summarize_repeats(candidate, expected)
    differences = [b - a for a, b in zip(left["repeat_scores"], right["repeat_scores"])]
    return {
        "baseline": left,
        "candidate": right,
        "paired_difference": {**_moments(differences), "values": differences},
        "mean_change": right["mean"] - left["mean"],
        "population_sd_change": right["sd_population"] - left["sd_population"],
        "worst_repeat_change": right["min"] - left["min"],
    }
