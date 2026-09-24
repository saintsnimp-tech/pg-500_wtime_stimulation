from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SurvivalEstimate:
    sample_size: int
    observed_events: int
    censored_cases: int
    p10_days: int | None
    median_days: int | None
    p90_days: int | None
    median_ci_low: int | None
    median_ci_high: int | None
    confidence_grade: str


def kaplan_meier_quantiles(rows: Iterable[tuple[int, int]]) -> tuple[int | None, int | None, int | None]:
    values = [(int(days), int(event)) for days, event in rows if days >= 0 and event in (0, 1)]
    if not values:
        return None, None, None
    survival = 1.0
    targets = [(0.90, None), (0.50, None), (0.10, None)]
    at_risk = len(values)
    for point in sorted({days for days, _ in values}):
        events = sum(event for days, event in values if days == point)
        censored = sum(1 - event for days, event in values if days == point)
        if events and at_risk:
            survival *= 1.0 - events / at_risk
            for index, (threshold, result) in enumerate(targets):
                if result is None and survival <= threshold:
                    targets[index] = (threshold, point)
        at_risk -= events + censored
    return targets[0][1], targets[1][1], targets[2][1]


def estimate_survival(
    rows: list[tuple[int, int]], bootstrap_runs: int = 400, seed: int = 500
) -> SurvivalEstimate:
    valid = [(days, event) for days, event in rows if 0 <= days <= 3650 and event in (0, 1)]
    p10, median, p90 = kaplan_meier_quantiles(valid)
    rng = random.Random(seed)
    medians: list[int] = []
    if valid:
        for _ in range(bootstrap_runs):
            sample = [valid[rng.randrange(len(valid))] for _ in valid]
            boot_median = kaplan_meier_quantiles(sample)[1]
            if boot_median is not None:
                medians.append(boot_median)
    medians.sort()
    low = medians[int(0.025 * (len(medians) - 1))] if medians else None
    high = medians[int(0.975 * (len(medians) - 1))] if medians else None
    events = sum(event for _, event in valid)
    censor_rate = 1.0 - events / len(valid) if valid else 1.0
    if len(valid) >= 300 and events >= 100 and censor_rate <= 0.70:
        grade = "B"
    elif len(valid) >= 100 and events >= 30:
        grade = "C"
    else:
        grade = "D"
    return SurvivalEstimate(
        sample_size=len(valid),
        observed_events=events,
        censored_cases=len(valid) - events,
        p10_days=p10,
        median_days=median,
        p90_days=p90,
        median_ci_low=low,
        median_ci_high=high,
        confidence_grade=grade,
    )

