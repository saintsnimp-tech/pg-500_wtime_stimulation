from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


MODEL_VERSION = "2.0.0"
DEFAULT_HALF_LIFE_DAYS = 540.0
DEFAULT_SHRINKAGE_PRIOR = 80.0


@dataclass(frozen=True)
class Observation:
    duration_days: float
    event_observed: int
    lodgement_date: date
    decision_date: date | None
    as_of_date: date
    education_level: str
    study_sector: str
    submit_location: str
    includes_partner: str
    is_diy: str
    provider_group: str
    field_of_study_group: str
    quality_score: int
    duplicate_key: str
    source_tier: str


@dataclass(frozen=True)
class WeightedPoint:
    duration_days: float
    event_observed: int
    weight: float


@dataclass(frozen=True)
class OfficialAnchor:
    stream: str
    p50_days: float
    p90_days: float
    snapshot_date: str
    page_updated: str
    source_url: str


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat((value or "")[:10])
    except ValueError:
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_observations(path: Path, min_quality: int = 40) -> list[Observation]:
    candidates: dict[str, Observation] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                duration = float(row["waiting_days"])
                event = int(row["event_observed"])
                quality = int(row["record_quality_score"])
            except (KeyError, ValueError):
                continue
            lodged = _parse_date(row.get("lodgement_date", ""))
            as_of = _parse_date(row.get("as_of_date", ""))
            decision = _parse_date(row.get("decision_date", ""))
            if quality < min_quality or lodged is None or as_of is None:
                continue
            if duration < 0 or duration > 3650 or event not in (0, 1):
                continue
            observation = Observation(
                duration_days=duration,
                event_observed=event,
                lodgement_date=lodged,
                decision_date=decision,
                as_of_date=as_of,
                education_level=row.get("education_level", "Unknown"),
                study_sector=row.get("study_sector", "Unknown"),
                submit_location=row.get("submit_location", "Unknown"),
                includes_partner=row.get("includes_partner", "unknown"),
                is_diy=row.get("is_diy", "unknown"),
                provider_group=row.get("provider_group", "Unknown"),
                field_of_study_group=row.get("field_of_study_group", "Unknown"),
                quality_score=quality,
                duplicate_key=row.get("duplicate_key", "") or row.get("record_id", ""),
                source_tier=row.get("source_quality_tier", "D"),
            )
            prior = candidates.get(observation.duplicate_key)
            if prior is None or observation.quality_score > prior.quality_score:
                candidates[observation.duplicate_key] = observation
    return list(candidates.values())


def load_official_anchors(path: Path) -> dict[str, OfficialAnchor]:
    anchors: dict[str, OfficialAnchor] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                p50 = float(row["percentile_50_days"])
                p90 = float(row["percentile_90_days"])
            except (KeyError, ValueError):
                continue
            anchor = OfficialAnchor(
                stream=row["stream"],
                p50_days=p50,
                p90_days=p90,
                snapshot_date=row.get("snapshot_date", ""),
                page_updated=row.get("official_page_updated", ""),
                source_url=row.get("source_url", ""),
            )
            anchors[anchor.stream] = anchor
    return anchors


def stream_for_sector(sector: str) -> str:
    mapping = {
        "Postgraduate Research": "Postgraduate Research Sector",
        "Higher Education": "Higher Education Sector",
        "Independent ELICOS": "Independent ELICOS Sector",
        "Schools": "Schools Sector",
        "Vocational Education and Training": "Vocational Education and Training Sector",
        "Non-Award": "Non-Award Sector",
        "Foreign Affairs or Defence": "Foreign Affairs or Defence Sector",
    }
    return mapping.get(sector, sector)


def weighted_points(
    observations: Iterable[Observation], reference_date: date, half_life_days: float = DEFAULT_HALF_LIFE_DAYS
) -> list[WeightedPoint]:
    tier_weight = {"A": 1.0, "B": 0.9, "B-C": 0.8, "C": 0.7, "D": 0.45}
    points: list[WeightedPoint] = []
    for item in observations:
        age_days = max(0, (reference_date - item.as_of_date).days)
        recency = 0.5 ** (age_days / half_life_days)
        quality = max(0.1, item.quality_score / 100.0)
        weight = recency * quality * tier_weight.get(item.source_tier, 0.5)
        points.append(WeightedPoint(item.duration_days, item.event_observed, weight))
    return points


def weighted_km_curve(points: Iterable[WeightedPoint]) -> list[tuple[float, float]]:
    grouped: dict[float, list[float]] = {}
    total_at_risk = 0.0
    for point in points:
        if point.duration_days < 0 or point.weight <= 0:
            continue
        values = grouped.setdefault(point.duration_days, [0.0, 0.0])
        values[0 if point.event_observed else 1] += point.weight
        total_at_risk += point.weight
    survival = 1.0
    curve = [(0.0, 1.0)]
    for duration in sorted(grouped):
        events, censored = grouped[duration]
        if events and total_at_risk > 0:
            survival *= max(0.0, 1.0 - events / total_at_risk)
            curve.append((duration, survival))
        total_at_risk -= events + censored
    return curve


def curve_quantile(curve: list[tuple[float, float]], probability: float) -> float | None:
    threshold = 1.0 - probability
    for duration, survival in curve:
        if survival <= threshold:
            return duration
    return None


def effective_sample_size(points: Iterable[WeightedPoint]) -> float:
    weights = [point.weight for point in points if point.weight > 0]
    denominator = sum(weight * weight for weight in weights)
    return (sum(weights) ** 2 / denominator) if denominator else 0.0


def _matches(item: Observation, filters: dict[str, str]) -> bool:
    return all(
        str(value or "").strip().lower() in {"", "unknown", "any"} or getattr(item, key) == value
        for key, value in filters.items()
    )


def _cohort_adjustment(
    all_observations: list[Observation], filters: dict[str, str], reference_date: date
) -> tuple[float, dict[str, Any], list[Observation], list[Observation]]:
    target = [item for item in all_observations if _matches(item, filters)]
    sector = filters.get("study_sector", "")
    broad = [item for item in all_observations if not sector or item.study_sector == sector]
    if len(broad) < 30:
        broad = all_observations
    target_points = weighted_points(target, reference_date)
    broad_points = weighted_points(broad, reference_date)
    target_median = curve_quantile(weighted_km_curve(target_points), 0.5)
    broad_median = curve_quantile(weighted_km_curve(broad_points), 0.5)
    n_eff = effective_sample_size(target_points)
    raw_ratio = 1.0
    if target_median and broad_median and broad_median > 0:
        raw_ratio = min(2.0, max(0.5, target_median / broad_median))
    shrinkage = n_eff / (n_eff + DEFAULT_SHRINKAGE_PRIOR)
    adjusted_ratio = math.exp(math.log(raw_ratio) * shrinkage)
    diagnostics = {
        "target_rows": len(target),
        "target_events": sum(item.event_observed for item in target),
        "effective_sample_size": round(n_eff, 1),
        "crowd_target_median_days": target_median,
        "crowd_broad_median_days": broad_median,
        "raw_crowd_ratio": round(raw_ratio, 4),
        "shrinkage_weight": round(shrinkage, 4),
        "official_anchor_multiplier": round(adjusted_ratio, 4),
    }
    return adjusted_ratio, diagnostics, target, broad


def _conditional_lognormal_quantile(
    probability: float, elapsed_days: float, median_days: float, p90_days: float
) -> float:
    normal = statistics.NormalDist()
    mu = math.log(max(0.1, median_days))
    sigma = max(0.05, (math.log(max(p90_days, median_days + 0.1)) - mu) / normal.inv_cdf(0.9))
    if elapsed_days <= 0:
        elapsed_cdf = 0.0
    else:
        elapsed_cdf = normal.cdf((math.log(max(elapsed_days, 0.01)) - mu) / sigma)
    conditional_cdf = min(0.999999, elapsed_cdf + probability * (1.0 - elapsed_cdf))
    return max(elapsed_days, math.exp(mu + sigma * normal.inv_cdf(conditional_cdf)))


def _parse_datetime(value: str, timezone_name: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
    return parsed


def _bootstrap_multiplier_interval(
    target: list[Observation], broad_median: float | None, reference_date: date, runs: int = 160
) -> tuple[float, float] | None:
    if len(target) < 20 or not broad_median:
        return None
    rng = random.Random(500)
    values: list[float] = []
    for _ in range(runs):
        sample = [target[rng.randrange(len(target))] for _ in target]
        points = weighted_points(sample, reference_date)
        sample_median = curve_quantile(weighted_km_curve(points), 0.5)
        if not sample_median:
            continue
        n_eff = effective_sample_size(points)
        raw_ratio = min(2.0, max(0.5, sample_median / broad_median))
        shrinkage = n_eff / (n_eff + DEFAULT_SHRINKAGE_PRIOR)
        values.append(math.exp(math.log(raw_ratio) * shrinkage))
    if len(values) < 20:
        return None
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def forecast(
    *,
    cases_path: Path,
    official_path: Path,
    lodged_at: str,
    as_of: str | None = None,
    timezone_name: str = "Asia/Shanghai",
    education_level: str = "Unknown",
    study_sector: str = "Unknown",
    submit_location: str = "Unknown",
    includes_partner: str = "unknown",
    is_diy: str = "unknown",
    provider_group: str = "Unknown",
    field_of_study_group: str = "Unknown",
    model_artifact: Path | None = None,
) -> dict[str, Any]:
    lodged = _parse_datetime(lodged_at, timezone_name)
    current = _parse_datetime(as_of, timezone_name) if as_of else datetime.now(lodged.tzinfo)
    if current < lodged:
        current = lodged
    elapsed_days = (current - lodged).total_seconds() / 86400.0
    observations = load_observations(cases_path)
    anchors = load_official_anchors(official_path)
    filters = {
        "education_level": education_level,
        "study_sector": study_sector,
        "submit_location": submit_location,
        "includes_partner": includes_partner,
        "is_diy": is_diy,
        "provider_group": provider_group,
        "field_of_study_group": field_of_study_group,
    }
    reference_date = max((item.as_of_date for item in observations), default=current.date())
    multiplier, diagnostics, target, broad = _cohort_adjustment(observations, filters, reference_date)
    anchor = anchors.get(stream_for_sector(study_sector))
    if anchor is None:
        raise ValueError(
            "No official day-valued p50/p90 anchor for this sector; choose a supported sector or use predict."
        )
    adjusted_p50 = anchor.p50_days * multiplier
    adjusted_p90 = anchor.p90_days * multiplier
    probabilities = (0.10, 0.50, 0.80, 0.90)
    total_days = {
        probability: _conditional_lognormal_quantile(probability, elapsed_days, adjusted_p50, adjusted_p90)
        for probability in probabilities
    }
    dates = {
        f"p{int(probability * 100)}": (lodged + timedelta(days=duration)).isoformat(timespec="minutes")
        for probability, duration in total_days.items()
    }
    remaining = {
        f"p{int(probability * 100)}": round(max(0.0, duration - elapsed_days), 1)
        for probability, duration in total_days.items()
    }
    broad_points = weighted_points(broad, reference_date)
    broad_median = curve_quantile(weighted_km_curve(broad_points), 0.5)
    multiplier_interval = _bootstrap_multiplier_interval(target, broad_median, reference_date)
    parameter_interval = None
    if multiplier_interval:
        parameter_interval = {
            "median_model_days_low": round(anchor.p50_days * multiplier_interval[0], 1),
            "median_model_days_high": round(anchor.p50_days * multiplier_interval[1], 1),
            "meaning": "95% bootstrap interval for crowd subgroup adjustment only; excludes source bias.",
        }

    latest_event = max((item.as_of_date for item in target), default=reference_date)
    recency_days = max(0, (reference_date - latest_event).days)
    official_score = 40
    sample_score = min(20, int(diagnostics["effective_sample_size"] / 10))
    event_score = min(10, diagnostics["target_events"] // 10)
    quality_score = min(10, round(sum(item.quality_score for item in target) / max(1, len(target)) / 10))
    recency_score = max(0, 10 - recency_days // 60)
    backtest_score = 0
    backtest = None
    if model_artifact and model_artifact.exists():
        artifact = json.loads(model_artifact.read_text(encoding="utf-8"))
        backtest = artifact.get("retrospective_backtest")
        if backtest is None and artifact.get("evaluations"):
            selected_name = artifact.get("selected_baseline")
            selected_evaluation = next(
                (
                    item
                    for item in artifact["evaluations"]
                    if item.get("model") == selected_name
                ),
                None,
            )
            backtest = {
                "time_split": artifact.get("time_split"),
                "selected_baseline": selected_name,
                "selected_evaluation": selected_evaluation,
                "role": "Independent benchmark only; the interactive forecast remains official-anchor based.",
            }
    # The retrospective diagnostic uses today's source registry and is not a true
    # historical replay, so it must not increase the production confidence score.
    source_bias_penalty = 15
    confidence_score = min(
        80,
        max(
            0,
            official_score + sample_score + event_score + quality_score + recency_score
            + backtest_score - source_bias_penalty,
        ),
    )
    confidence_grade = "A" if confidence_score >= 85 else "B" if confidence_score >= 70 else "C" if confidence_score >= 50 else "D"
    return {
        "model_version": MODEL_VERSION,
        "lodged_at": lodged.isoformat(timespec="minutes"),
        "as_of": current.isoformat(timespec="minutes"),
        "elapsed_days": round(elapsed_days, 3),
        "predicted_decision_at": dates,
        "remaining_days": remaining,
        "predictive_interval": {
            "level": "80%",
            "low": dates["p10"],
            "high": dates["p90"],
            "meaning": "Model outcome distribution, not a confidence interval or guarantee.",
        },
        "median_parameter_uncertainty": parameter_interval,
        "data_confidence": {
            "score": confidence_score,
            "grade": confidence_grade,
            "meaning": "Evidence sufficiency score; not the probability that the visa will be granted.",
            "components": {
                "official_anchor": official_score,
                "effective_sample": sample_score,
                "observed_events": event_score,
                "row_quality": quality_score,
                "recency": recency_score,
                "backtest_calibration": backtest_score,
                "self_selection_bias_penalty": -source_bias_penalty,
            },
        },
        "official_anchor": {
            **asdict(anchor),
            "adjusted_p50_days": round(adjusted_p50, 2),
            "adjusted_p90_days": round(adjusted_p90, 2),
        },
        "crowd_adjustment": diagnostics,
        "filters": filters,
        "backtest": backtest,
        "limitations": [
            "Official percentiles describe recently decided applications and are not a service promise.",
            "Crowd data is self-selected and dominated by postgraduate research cases.",
            "Training timestamps are mostly date-level; minute precision preserves the input clock time but not minute-level model resolution.",
            "A predicted decision date is not a prediction of grant versus refusal.",
        ],
    }


def _retrospective_backtest(observations: list[Observation], reference_date: date) -> dict[str, Any]:
    cutoff = date(2025, 12, 31)
    train: list[Observation] = []
    tests: list[Observation] = []
    for item in observations:
        if item.lodgement_date > cutoff:
            continue
        if item.event_observed and item.decision_date and item.decision_date <= cutoff:
            train.append(item)
        else:
            duration = float((cutoff - item.lodgement_date).days)
            train.append(replace(item, duration_days=duration, event_observed=0, decision_date=None, as_of_date=cutoff))
            if item.event_observed and item.decision_date and item.decision_date > cutoff:
                tests.append(item)
    curve = weighted_km_curve(weighted_points(train, cutoff))
    p10 = curve_quantile(curve, 0.10)
    p50 = curve_quantile(curve, 0.50)
    p90 = curve_quantile(curve, 0.90)
    if not tests or p10 is None or p50 is None or p90 is None:
        return {"cutoff": cutoff.isoformat(), "test_events": len(tests), "status": "insufficient"}
    actual = [item.duration_days for item in tests]
    coverage = sum(p10 <= value <= p90 for value in actual) / len(actual)
    absolute_errors = sorted(abs(value - p50) for value in actual)
    median_error = absolute_errors[len(absolute_errors) // 2]
    return {
        "cutoff": cutoff.isoformat(),
        "test_events": len(tests),
        "p10_days": round(p10, 1),
        "p50_days": round(p50, 1),
        "p90_days": round(p90, 1),
        "p10_p90_coverage": round(coverage, 4),
        "median_absolute_error_days": round(median_error, 1),
        "warning": "Retrospective pseudo-backtest; source-record availability at the cutoff is not observable.",
    }


def train_model(cases_path: Path, official_path: Path, output: Path) -> dict[str, Any]:
    observations = load_observations(cases_path)
    anchors = load_official_anchors(official_path)
    reference_date = max((item.as_of_date for item in observations), default=date.today())
    artifact = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_snapshot_date": reference_date.isoformat(),
        "case_dataset_sha256": file_sha256(cases_path),
        "official_dataset_sha256": file_sha256(official_path),
        "training_rows_after_quality_and_dedup": len(observations),
        "observed_events": sum(item.event_observed for item in observations),
        "right_censored": sum(1 - item.event_observed for item in observations),
        "hyperparameters": {
            "recency_half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "cohort_shrinkage_prior_effective_n": DEFAULT_SHRINKAGE_PRIOR,
            "crowd_ratio_cap": [0.5, 2.0],
            "quality_threshold": 40,
        },
        "official_anchors": {name: asdict(anchor) for name, anchor in anchors.items()},
        "retrospective_backtest": _retrospective_backtest(observations, reference_date),
        "method": "Official-anchored hierarchical weighted survival model with right censoring and recency weighting",
        "known_biases": [
            "Self-selected crowd sample",
            "Postgraduate research over-representation",
            "Historical source availability cannot be reconstructed for a true time-split backtest",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact
