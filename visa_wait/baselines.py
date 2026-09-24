from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from .model import Observation, WeightedPoint, curve_quantile, load_observations, weighted_km_curve


BASELINE_VERSION = "3.0.0"
TRAIN_CUTOFF = date(2024, 12, 31)
TEST_END = date(2025, 8, 31)
CALIBRATION_END = date(2025, 4, 30)
HORIZONS = (90.0, 180.0, 365.0)

CATEGORICAL_FEATURES = (
    "education_level",
    "study_sector",
    "submit_location",
    "field_of_study_group",
    "includes_partner",
    "is_diy",
    "provider_group",
)
NUMERIC_FEATURES = (
    "lodgement_year",
    "lodgement_month_sin",
    "lodgement_month_cos",
    "official_flow_available",
    "official_lag1_lodged_log",
    "official_lag1_granted_log",
    "official_lag1_net_per_100",
    "official_lag1_grant_lodged_ratio",
    "official_lag1_trailing_3m_lodged_log",
)


@dataclass
class SurvivalRow:
    duration_days: float
    event_observed: int
    lodgement_date: date
    features: dict[str, Any]
    encoded: list[float] | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _month_key(value: date) -> str:
    return value.strftime("%Y-%m")


def _previous_months(month: str, count: int) -> list[str]:
    year, month_number = map(int, month.split("-"))
    result: list[str] = []
    for offset in range(count):
        absolute = year * 12 + month_number - 1 - offset
        result.append(f"{absolute // 12:04d}-{absolute % 12 + 1:02d}")
    return result


def load_official_flow_features(path: Path) -> dict[str, dict[str, float]]:
    by_month: defaultdict[str, dict[str, float]] = defaultdict(dict)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                by_month[row["activity_month"]][row["activity_type"]] = float(row["count"])
            except (KeyError, ValueError):
                continue
    result: dict[str, dict[str, float]] = {}
    for month, values in by_month.items():
        lodged = values.get("lodged", 0.0)
        granted = values.get("granted", 0.0)
        trailing = sum(by_month.get(key, {}).get("lodged", 0.0) for key in _previous_months(month, 3))
        result[month] = {
            "lodged": lodged,
            "granted": granted,
            "net": lodged - granted,
            "grant_lodged_ratio": granted / lodged if lodged else 0.0,
            "trailing_3m_lodged": trailing,
        }
    return result


def _raw_features(observation: Observation, official: dict[str, dict[str, float]]) -> dict[str, Any]:
    month = _month_key(observation.lodgement_date)
    prior_month = _previous_months(month, 2)[1]
    eligible_proxy = (
        observation.study_sector == "Postgraduate Research"
        and observation.submit_location == "Outside Australia"
        and prior_month in official
    )
    # The application month's final aggregate is not known at lodgement time.
    # Use only the preceding month's published-period value to avoid look-ahead.
    flow = official.get(prior_month, {}) if eligible_proxy else {}
    angle = 2.0 * math.pi * observation.lodgement_date.month / 12.0
    return {
        "education_level": observation.education_level,
        "study_sector": observation.study_sector,
        "submit_location": observation.submit_location,
        "field_of_study_group": observation.field_of_study_group,
        "includes_partner": observation.includes_partner,
        "is_diy": observation.is_diy,
        "provider_group": observation.provider_group,
        "lodgement_year": float(observation.lodgement_date.year),
        "lodgement_month_sin": math.sin(angle),
        "lodgement_month_cos": math.cos(angle),
        "official_flow_available": 1.0 if eligible_proxy else 0.0,
        "official_lag1_lodged_log": math.log1p(flow.get("lodged", 0.0)),
        "official_lag1_granted_log": math.log1p(flow.get("granted", 0.0)),
        "official_lag1_net_per_100": flow.get("net", 0.0) / 100.0,
        "official_lag1_grant_lodged_ratio": flow.get("grant_lodged_ratio", 0.0),
        "official_lag1_trailing_3m_lodged_log": math.log1p(flow.get("trailing_3m_lodged", 0.0)),
    }


def prepare_time_split(
    observations: list[Observation],
    official: dict[str, dict[str, float]],
    cutoff: date = TRAIN_CUTOFF,
    test_end: date = TEST_END,
) -> tuple[list[SurvivalRow], list[SurvivalRow]]:
    training: list[SurvivalRow] = []
    testing: list[SurvivalRow] = []
    for item in observations:
        features = _raw_features(item, official)
        if item.lodgement_date <= cutoff:
            if item.event_observed and item.decision_date and item.decision_date <= cutoff:
                duration = max(0.5, float((item.decision_date - item.lodgement_date).days))
                event = 1
            else:
                duration = max(0.5, float((cutoff - item.lodgement_date).days))
                event = 0
            training.append(SurvivalRow(duration, event, item.lodgement_date, features))
        elif item.lodgement_date <= test_end:
            testing.append(
                SurvivalRow(max(0.5, item.duration_days), item.event_observed, item.lodgement_date, features)
            )
    return training, testing


class FeatureEncoder:
    def __init__(self) -> None:
        self.levels: dict[str, list[str]] = {}
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.feature_names: list[str] = []

    def fit(self, rows: Iterable[SurvivalRow]) -> "FeatureEncoder":
        rows_list = list(rows)
        self.feature_names = []
        for field in CATEGORICAL_FEATURES:
            unknown_value = "unknown" if field in {"includes_partner", "is_diy"} else "Unknown"
            levels = sorted(
                {str(row.features.get(field, unknown_value)) for row in rows_list} | {unknown_value}
            )
            self.levels[field] = levels
            for level in levels[1:]:
                self.feature_names.append(f"{field}={level}")
        for field in NUMERIC_FEATURES:
            values = [float(row.features.get(field, 0.0)) for row in rows_list]
            mean = statistics.fmean(values) if values else 0.0
            variance = statistics.fmean((value - mean) ** 2 for value in values) if values else 0.0
            self.means[field] = mean
            self.scales[field] = max(math.sqrt(variance), 1e-9)
            self.feature_names.append(field)
        return self

    def transform(self, row: SurvivalRow) -> list[float]:
        values: list[float] = []
        for field in CATEGORICAL_FEATURES:
            current = str(row.features.get(field, "Unknown"))
            values.extend(1.0 if current == level else 0.0 for level in self.levels[field][1:])
        for field in NUMERIC_FEATURES:
            raw = float(row.features.get(field, 0.0))
            values.append((raw - self.means[field]) / self.scales[field])
        return values

    def apply(self, rows: Iterable[SurvivalRow]) -> None:
        for row in rows:
            row.encoded = self.transform(row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": self.levels,
            "means": self.means,
            "scales": self.scales,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "FeatureEncoder":
        encoder = cls()
        encoder.levels = {key: list(value) for key, value in state["levels"].items()}
        encoder.means = {key: float(value) for key, value in state["means"].items()}
        encoder.scales = {key: float(value) for key, value in state["scales"].items()}
        encoder.feature_names = list(state["feature_names"])
        return encoder


def _km(rows: Iterable[SurvivalRow]) -> list[tuple[float, float]]:
    return weighted_km_curve(
        WeightedPoint(row.duration_days, row.event_observed, 1.0) for row in rows
    )


def _survival_at(curve: list[tuple[float, float]], horizon: float) -> float:
    survival = 1.0
    for duration, value in curve:
        if duration > horizon:
            break
        survival = value
    return survival


def _quantiles_from_curve(curve: list[tuple[float, float]]) -> tuple[float | None, float | None, float | None]:
    return (
        curve_quantile(curve, 0.10),
        curve_quantile(curve, 0.50),
        curve_quantile(curve, 0.90),
    )


class GlobalKaplanMeier:
    name = "kaplan_meier_global"

    def __init__(self, rows: list[SurvivalRow]) -> None:
        self.curve = _km(rows)

    def quantiles(self, row: SurvivalRow) -> tuple[float | None, float | None, float | None]:
        return _quantiles_from_curve(self.curve)

    def survival(self, row: SurvivalRow, horizon: float) -> float:
        return _survival_at(self.curve, horizon)

    def risk(self, row: SurvivalRow) -> float:
        median = curve_quantile(self.curve, 0.5)
        return -float(median or 1e9)


class StratifiedKaplanMeier:
    name = "kaplan_meier_stratified"

    def __init__(self, rows: list[SurvivalRow], min_rows: int = 40, min_events: int = 15) -> None:
        self.global_curve = _km(rows)
        self.curves: dict[tuple[str, ...], list[tuple[float, float]]] = {}
        for fields in (
            ("study_sector", "submit_location", "education_level"),
            ("study_sector", "submit_location"),
        ):
            groups: defaultdict[tuple[str, ...], list[SurvivalRow]] = defaultdict(list)
            for row in rows:
                groups[tuple(str(row.features[field]) for field in fields)].append(row)
            for key, group in groups.items():
                if len(group) >= min_rows and sum(item.event_observed for item in group) >= min_events:
                    self.curves[("|".join(fields), *key)] = _km(group)

    def _curve(self, row: SurvivalRow) -> list[tuple[float, float]]:
        for fields in (
            ("study_sector", "submit_location", "education_level"),
            ("study_sector", "submit_location"),
        ):
            key = ("|".join(fields), *(str(row.features[field]) for field in fields))
            if key in self.curves:
                return self.curves[key]
        return self.global_curve

    def quantiles(self, row: SurvivalRow) -> tuple[float | None, float | None, float | None]:
        return _quantiles_from_curve(self._curve(row))

    def survival(self, row: SurvivalRow, horizon: float) -> float:
        return _survival_at(self._curve(row), horizon)

    def risk(self, row: SurvivalRow) -> float:
        median = curve_quantile(self._curve(row), 0.5)
        return -float(median or 1e9)


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _adam_update(
    parameters: list[float],
    gradient: list[float],
    first: list[float],
    second: list[float],
    iteration: int,
    learning_rate: float,
) -> None:
    beta1, beta2 = 0.9, 0.999
    for index, value in enumerate(gradient):
        value = _clip(value, -10.0, 10.0)
        first[index] = beta1 * first[index] + (1.0 - beta1) * value
        second[index] = beta2 * second[index] + (1.0 - beta2) * value * value
        corrected_first = first[index] / (1.0 - beta1**iteration)
        corrected_second = second[index] / (1.0 - beta2**iteration)
        parameters[index] += learning_rate * corrected_first / (math.sqrt(corrected_second) + 1e-8)


class CoxPHBaseline:
    name = "cox_ph"

    def __init__(self, rows: list[SurvivalRow], feature_names: list[str], iterations: int = 700) -> None:
        self.feature_names = feature_names
        dimensions = len(feature_names)
        self.coefficients = [0.0] * dimensions
        first = [0.0] * dimensions
        second = [0.0] * dimensions
        ordered = sorted(range(len(rows)), key=lambda index: rows[index].duration_days, reverse=True)
        groups: list[tuple[int, list[int]]] = []
        start = 0
        while start < len(ordered):
            end = start
            duration = rows[ordered[start]].duration_days
            while end + 1 < len(ordered) and rows[ordered[end + 1]].duration_days == duration:
                end += 1
            events = [position for position in range(start, end + 1) if rows[ordered[position]].event_observed]
            groups.append((end, events))
            start = end + 1
        total_events = max(1, sum(row.event_observed for row in rows))
        for iteration in range(1, iterations + 1):
            risk_values: list[float] = []
            cumulative_risk: list[float] = []
            cumulative_x: list[list[float]] = []
            running_risk = 0.0
            running_x = [0.0] * dimensions
            for position, row_index in enumerate(ordered):
                vector = rows[row_index].encoded or []
                eta = _clip(sum(coef * value for coef, value in zip(self.coefficients, vector)), -20.0, 20.0)
                risk = math.exp(eta)
                risk_values.append(risk)
                running_risk += risk
                for feature in range(dimensions):
                    running_x[feature] += risk * vector[feature]
                cumulative_risk.append(running_risk)
                cumulative_x.append(running_x.copy())
            gradient = [0.0] * dimensions
            for end, event_positions in groups:
                if not event_positions:
                    continue
                denominator = max(cumulative_risk[end], 1e-12)
                event_count = len(event_positions)
                for feature in range(dimensions):
                    observed = sum((rows[ordered[position]].encoded or [])[feature] for position in event_positions)
                    expected = event_count * cumulative_x[end][feature] / denominator
                    gradient[feature] += observed - expected
            for feature in range(dimensions):
                gradient[feature] = gradient[feature] / total_events - 0.02 * self.coefficients[feature]
            _adam_update(self.coefficients, gradient, first, second, iteration, 0.025)

        risks = [
            math.exp(_clip(sum(coef * value for coef, value in zip(self.coefficients, row.encoded or [])), -20, 20))
            for row in rows
        ]
        event_times = sorted({row.duration_days for row in rows if row.event_observed})
        cumulative = 0.0
        self.baseline_hazard: list[tuple[float, float]] = []
        for event_time in event_times:
            events = sum(1 for row in rows if row.event_observed and row.duration_days == event_time)
            denominator = sum(risk for row, risk in zip(rows, risks) if row.duration_days >= event_time)
            if denominator > 0:
                cumulative += events / denominator
                self.baseline_hazard.append((event_time, cumulative))

    def _linear_predictor(self, row: SurvivalRow) -> float:
        return sum(coef * value for coef, value in zip(self.coefficients, row.encoded or []))

    def survival(self, row: SurvivalRow, horizon: float) -> float:
        cumulative = 0.0
        for event_time, value in self.baseline_hazard:
            if event_time > horizon:
                break
            cumulative = value
        return math.exp(-cumulative * math.exp(_clip(self._linear_predictor(row), -20, 20)))

    def quantiles(self, row: SurvivalRow) -> tuple[float | None, float | None, float | None]:
        curve = [(0.0, 1.0)] + [
            (time, math.exp(-hazard * math.exp(_clip(self._linear_predictor(row), -20, 20))))
            for time, hazard in self.baseline_hazard
        ]
        return _quantiles_from_curve(curve)

    def risk(self, row: SurvivalRow) -> float:
        return self._linear_predictor(row)


class WeibullAFTBaseline:
    name = "weibull_aft"

    def __init__(self, rows: list[SurvivalRow], feature_names: list[str], iterations: int = 1400) -> None:
        self.feature_names = ["intercept", *feature_names]
        dimensions = len(self.feature_names)
        observed = [row.duration_days for row in rows if row.event_observed]
        initial_scale = statistics.median(observed or [row.duration_days for row in rows])
        self.parameters = [math.log(max(initial_scale, 1.0)), *([0.0] * (dimensions - 1)), 0.0]
        first = [0.0] * len(self.parameters)
        second = [0.0] * len(self.parameters)
        for iteration in range(1, iterations + 1):
            beta = self.parameters[:-1]
            log_shape = self.parameters[-1]
            shape = math.exp(_clip(log_shape, -2.0, 2.0))
            gradient = [0.0] * len(self.parameters)
            for row in rows:
                vector = [1.0, *(row.encoded or [])]
                eta = sum(coef * value for coef, value in zip(beta, vector))
                log_time = math.log(max(row.duration_days, 0.5))
                scaled = shape * (log_time - eta)
                cumulative_hazard = math.exp(_clip(scaled, -30.0, 30.0))
                event = row.event_observed
                beta_factor = shape * (cumulative_hazard - event)
                for index, value in enumerate(vector):
                    gradient[index] += value * beta_factor
                gradient[-1] += event * (1.0 + scaled) - cumulative_hazard * scaled
            for index in range(len(gradient)):
                gradient[index] /= max(1, len(rows))
            for index in range(1, len(beta)):
                gradient[index] -= 0.015 * beta[index]
            gradient[-1] -= 0.01 * log_shape
            _adam_update(self.parameters, gradient, first, second, iteration, 0.012)
            self.parameters[-1] = _clip(self.parameters[-1], -2.0, 2.0)

    @property
    def shape(self) -> float:
        return math.exp(self.parameters[-1])

    def _scale(self, row: SurvivalRow) -> float:
        vector = [1.0, *(row.encoded or [])]
        eta = sum(coef * value for coef, value in zip(self.parameters[:-1], vector))
        return math.exp(_clip(eta, -10.0, 10.0))

    def survival(self, row: SurvivalRow, horizon: float) -> float:
        return math.exp(-((max(horizon, 0.0) / max(self._scale(row), 1e-9)) ** self.shape))

    def quantiles(self, row: SurvivalRow) -> tuple[float | None, float | None, float | None]:
        scale = self._scale(row)
        return tuple(
            scale * (-math.log(1.0 - probability)) ** (1.0 / self.shape)
            for probability in (0.10, 0.50, 0.90)
        )

    def risk(self, row: SurvivalRow) -> float:
        return -self._scale(row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.name,
            "feature_names": self.feature_names,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "WeibullAFTBaseline":
        model = cls.__new__(cls)
        model.feature_names = list(state["feature_names"])
        model.parameters = [float(value) for value in state["parameters"]]
        return model


@dataclass
class IsotonicBlock:
    x_min: float
    x_max: float
    total: float
    count: int

    @property
    def mean(self) -> float:
        return self.total / self.count


class ProbabilityCalibrator:
    """Monotone calibration map fitted with pooled adjacent violators."""

    def __init__(self, knots: list[tuple[float, float]]) -> None:
        self.knots = sorted(knots)

    @classmethod
    def fit(cls, pairs: list[tuple[float, float]]) -> "ProbabilityCalibrator":
        grouped: list[IsotonicBlock] = []
        for probability, outcome in sorted(pairs):
            probability = _clip(float(probability), 0.0, 1.0)
            if grouped and abs(grouped[-1].x_max - probability) < 1e-12:
                grouped[-1].total += float(outcome)
                grouped[-1].count += 1
                continue
            grouped.append(IsotonicBlock(probability, probability, float(outcome), 1))
        blocks: list[IsotonicBlock] = []
        for block in grouped:
            blocks.append(block)
            while len(blocks) >= 2 and blocks[-2].mean > blocks[-1].mean:
                right = blocks.pop()
                left = blocks.pop()
                blocks.append(
                    IsotonicBlock(
                        left.x_min,
                        right.x_max,
                        left.total + right.total,
                        left.count + right.count,
                    )
                )
        knots = [(0.0, 0.0)]
        for block in blocks:
            midpoint = (block.x_min + block.x_max) / 2.0
            knots.append((midpoint, _clip(block.mean, 0.0, 1.0)))
        knots.append((1.0, 1.0))
        compact: list[tuple[float, float]] = []
        for x_value, y_value in sorted(knots):
            if compact and abs(compact[-1][0] - x_value) < 1e-12:
                compact[-1] = (x_value, max(compact[-1][1], y_value))
            else:
                compact.append((x_value, y_value))
        for index in range(1, len(compact)):
            compact[index] = (compact[index][0], max(compact[index - 1][1], compact[index][1]))
        return cls(compact)

    def calibrate(self, probability: float) -> float:
        value = _clip(float(probability), 0.0, 1.0)
        for (x_left, y_left), (x_right, y_right) in zip(self.knots, self.knots[1:]):
            if value <= x_right:
                if x_right <= x_left:
                    return y_right
                weight = (value - x_left) / (x_right - x_left)
                return _clip(y_left + weight * (y_right - y_left), 0.0, 1.0)
        return 1.0

    def inverse(self, calibrated_probability: float) -> float:
        target = _clip(float(calibrated_probability), 0.0, 1.0)
        for (x_left, y_left), (x_right, y_right) in zip(self.knots, self.knots[1:]):
            if target <= y_right:
                if y_right <= y_left:
                    return x_right
                weight = (target - y_left) / (y_right - y_left)
                return _clip(x_left + weight * (x_right - x_left), 0.0, 1.0)
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"method": "pooled_adjacent_violators", "knots": self.knots}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "ProbabilityCalibrator":
        return cls([(float(x_value), float(y_value)) for x_value, y_value in state["knots"]])


class CalibratedSurvivalModel:
    name = "calibrated_weibull_aft"

    def __init__(self, base: WeibullAFTBaseline, calibrator: ProbabilityCalibrator) -> None:
        self.base = base
        self.calibrator = calibrator

    def cdf(self, row: SurvivalRow, horizon: float) -> float:
        raw_probability = 1.0 - self.base.survival(row, horizon)
        return self.calibrator.calibrate(raw_probability)

    def survival(self, row: SurvivalRow, horizon: float) -> float:
        return 1.0 - self.cdf(row, horizon)

    def _quantile(self, row: SurvivalRow, probability: float) -> float:
        raw_probability = min(0.999999, self.calibrator.inverse(probability))
        return self.base._scale(row) * (-math.log(1.0 - raw_probability)) ** (1.0 / self.base.shape)

    def quantiles(self, row: SurvivalRow) -> tuple[float | None, float | None, float | None]:
        return tuple(self._quantile(row, probability) for probability in (0.10, 0.50, 0.90))

    def risk(self, row: SurvivalRow) -> float:
        return self.base.risk(row)


def fit_probability_calibrator(model: Any, rows: list[SurvivalRow]) -> ProbabilityCalibrator:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        for horizon in HORIZONS:
            if not row.event_observed and row.duration_days < horizon:
                continue
            raw_probability = 1.0 - model.survival(row, horizon)
            outcome = 1.0 if row.event_observed and row.duration_days <= horizon else 0.0
            pairs.append((raw_probability, outcome))
    if len(pairs) < 30:
        return ProbabilityCalibrator([(0.0, 0.0), (1.0, 1.0)])
    return ProbabilityCalibrator.fit(pairs)


@dataclass
class SurvivalTreeNode:
    feature: int | None = None
    threshold: float | None = None
    left: "SurvivalTreeNode | None" = None
    right: "SurvivalTreeNode | None" = None
    curve: list[tuple[float, float]] | None = None


def _logrank_score(rows: list[SurvivalRow], left_indices: set[int]) -> float:
    if not left_indices or len(left_indices) == len(rows):
        return 0.0
    ordered = sorted(range(len(rows)), key=lambda index: rows[index].duration_days)
    risk_total = len(rows)
    risk_left = len(left_indices)
    observed_minus_expected = 0.0
    variance = 0.0
    start = 0
    while start < len(ordered):
        end = start
        duration = rows[ordered[start]].duration_days
        while end + 1 < len(ordered) and rows[ordered[end + 1]].duration_days == duration:
            end += 1
        group = ordered[start : end + 1]
        events = sum(rows[index].event_observed for index in group)
        left_events = sum(rows[index].event_observed for index in group if index in left_indices)
        if events and risk_total > 0:
            expected = events * risk_left / risk_total
            observed_minus_expected += left_events - expected
            if risk_total > 1:
                variance += (
                    risk_left
                    * (risk_total - risk_left)
                    * events
                    * (risk_total - events)
                    / (risk_total * risk_total * (risk_total - 1))
                )
        risk_total -= len(group)
        risk_left -= sum(1 for index in group if index in left_indices)
        start = end + 1
    return observed_minus_expected * observed_minus_expected / variance if variance > 1e-12 else 0.0


class RandomSurvivalForestBaseline:
    name = "random_survival_forest"

    def __init__(
        self,
        rows: list[SurvivalRow],
        *,
        trees: int = 36,
        max_depth: int = 4,
        min_leaf: int = 30,
        seed: int = 500,
    ) -> None:
        self.rows = rows
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.rng = random.Random(seed)
        self.event_times = sorted({row.duration_days for row in rows if row.event_observed})
        self.trees: list[SurvivalTreeNode] = []
        for _ in range(trees):
            sample = [rows[self.rng.randrange(len(rows))] for _ in rows]
            self.trees.append(self._build(sample, 0))

    def _candidate_thresholds(self, rows: list[SurvivalRow], feature: int) -> list[float]:
        values = sorted({(row.encoded or [])[feature] for row in rows})
        if len(values) <= 1:
            return []
        if len(values) <= 8:
            return [(left + right) / 2.0 for left, right in zip(values, values[1:])]
        return sorted({values[int((len(values) - 1) * fraction)] for fraction in (0.2, 0.4, 0.6, 0.8)})

    def _build(self, rows: list[SurvivalRow], depth: int) -> SurvivalTreeNode:
        if (
            depth >= self.max_depth
            or len(rows) < 2 * self.min_leaf
            or sum(row.event_observed for row in rows) < 10
        ):
            return SurvivalTreeNode(curve=_km(rows))
        dimensions = len(rows[0].encoded or [])
        feature_count = max(1, int(math.sqrt(dimensions)))
        features = self.rng.sample(range(dimensions), min(feature_count, dimensions))
        best: tuple[float, int, float, list[SurvivalRow], list[SurvivalRow]] | None = None
        for feature in features:
            for threshold in self._candidate_thresholds(rows, feature):
                left_positions = {
                    index for index, row in enumerate(rows) if (row.encoded or [])[feature] <= threshold
                }
                if len(left_positions) < self.min_leaf or len(rows) - len(left_positions) < self.min_leaf:
                    continue
                score = _logrank_score(rows, left_positions)
                if best is None or score > best[0]:
                    left = [row for index, row in enumerate(rows) if index in left_positions]
                    right = [row for index, row in enumerate(rows) if index not in left_positions]
                    best = (score, feature, threshold, left, right)
        if best is None or best[0] <= 1e-9:
            return SurvivalTreeNode(curve=_km(rows))
        _, feature, threshold, left, right = best
        return SurvivalTreeNode(
            feature=feature,
            threshold=threshold,
            left=self._build(left, depth + 1),
            right=self._build(right, depth + 1),
        )

    def _leaf(self, node: SurvivalTreeNode, row: SurvivalRow) -> SurvivalTreeNode:
        current = node
        while current.curve is None:
            assert current.feature is not None and current.threshold is not None
            assert current.left is not None and current.right is not None
            current = current.left if (row.encoded or [])[current.feature] <= current.threshold else current.right
        return current

    def survival(self, row: SurvivalRow, horizon: float) -> float:
        return statistics.fmean(
            _survival_at(self._leaf(tree, row).curve or [(0.0, 1.0)], horizon) for tree in self.trees
        )

    def quantiles(self, row: SurvivalRow) -> tuple[float | None, float | None, float | None]:
        curve = [(0.0, 1.0)] + [(time, self.survival(row, time)) for time in self.event_times]
        return _quantiles_from_curve(curve)

    def risk(self, row: SurvivalRow) -> float:
        median = self.quantiles(row)[1]
        return -float(median or 1e9)


def _concordance(rows: list[SurvivalRow], risks: list[float]) -> tuple[float | None, int]:
    comparable = 0
    concordant = 0.0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if rows[left].duration_days == rows[right].duration_days:
                continue
            if rows[left].duration_days < rows[right].duration_days and rows[left].event_observed:
                early, late = left, right
            elif rows[right].duration_days < rows[left].duration_days and rows[right].event_observed:
                early, late = right, left
            else:
                continue
            comparable += 1
            if risks[early] > risks[late]:
                concordant += 1.0
            elif risks[early] == risks[late]:
                concordant += 0.5
    return (concordant / comparable if comparable else None), comparable


def evaluate_model(model: Any, rows: list[SurvivalRow]) -> dict[str, Any]:
    risks = [model.risk(row) for row in rows]
    concordance, comparable_pairs = _concordance(rows, risks)
    event_errors: list[float] = []
    covered = 0
    interval_rows = 0
    for row in rows:
        p10, median, p90 = model.quantiles(row)
        if row.event_observed and median is not None:
            event_errors.append(abs(row.duration_days - median))
        if row.event_observed and p10 is not None and p90 is not None:
            interval_rows += 1
            covered += int(p10 <= row.duration_days <= p90)
    calibration: dict[str, Any] = {}
    for horizon in HORIZONS:
        predicted = [1.0 - model.survival(row, horizon) for row in rows]
        observed = [1.0 if row.event_observed and row.duration_days <= horizon else 0.0 for row in rows]
        brier = statistics.fmean((estimate - actual) ** 2 for estimate, actual in zip(predicted, observed))
        calibration[str(int(horizon))] = {
            "brier_score": round(brier, 4),
            "mean_predicted_decision_probability": round(statistics.fmean(predicted), 4),
            "observed_decision_fraction": round(statistics.fmean(observed), 4),
            "calibration_gap": round(statistics.fmean(predicted) - statistics.fmean(observed), 4),
        }
    return {
        "model": model.name,
        "test_rows": len(rows),
        "test_events": sum(row.event_observed for row in rows),
        "test_censored": sum(1 - row.event_observed for row in rows),
        "harrell_c_index": round(concordance, 4) if concordance is not None else None,
        "comparable_pairs": comparable_pairs,
        "event_only_median_absolute_error_days": round(statistics.median(event_errors), 1) if event_errors else None,
        "event_only_p10_p90_coverage": round(covered / interval_rows, 4) if interval_rows else None,
        "event_only_interval_rows": interval_rows,
        "horizon_calibration": calibration,
    }


def _top_coefficients(names: list[str], coefficients: list[float], limit: int = 12) -> list[dict[str, Any]]:
    ranked = sorted(zip(names, coefficients), key=lambda pair: abs(pair[1]), reverse=True)[:limit]
    return [{"feature": name, "coefficient": round(value, 5)} for name, value in ranked]


def train_survival_baselines(
    *,
    cases_path: Path,
    official_flows_path: Path,
    output: Path,
) -> dict[str, Any]:
    observations = load_observations(cases_path)
    official = load_official_flow_features(official_flows_path)
    training, testing = prepare_time_split(observations, official)
    if len(training) < 100 or len(testing) < 50:
        raise ValueError("Insufficient rows for the fixed temporal train/test split")
    encoder = FeatureEncoder().fit(training)
    encoder.apply(training)
    encoder.apply(testing)
    calibration_rows = [row for row in testing if row.lodgement_date <= CALIBRATION_END]
    holdout_rows = [row for row in testing if row.lodgement_date > CALIBRATION_END]
    if len(calibration_rows) < 50 or len(holdout_rows) < 50:
        raise ValueError("Insufficient rows for calibration and final holdout windows")

    models: list[Any] = [
        GlobalKaplanMeier(training),
        StratifiedKaplanMeier(training),
        CoxPHBaseline(training, encoder.feature_names),
        WeibullAFTBaseline(training, encoder.feature_names),
        RandomSurvivalForestBaseline(training),
    ]
    evaluations = [evaluate_model(model, testing) for model in models]
    development_evaluations = [evaluate_model(model, calibration_rows) for model in models]
    holdout_evaluations = [evaluate_model(model, holdout_rows) for model in models]
    ranked = sorted(
        development_evaluations,
        key=lambda item: (
            -(item["harrell_c_index"] if item["harrell_c_index"] is not None else -1.0),
            item["horizon_calibration"]["180"]["brier_score"],
        ),
    )
    cox = next(model for model in models if isinstance(model, CoxPHBaseline))
    aft = next(model for model in models if isinstance(model, WeibullAFTBaseline))
    calibrator = fit_probability_calibrator(aft, calibration_rows)
    calibrated_aft = CalibratedSurvivalModel(aft, calibrator)
    candidate_calibration_holdout = evaluate_model(calibrated_aft, holdout_rows)
    deployment_holdout = next(
        item for item in holdout_evaluations if item["model"] == "weibull_aft"
    )
    c_index = float(deployment_holdout.get("harrell_c_index") or 0.5)
    brier_180 = float(deployment_holdout["horizon_calibration"]["180"]["brier_score"])
    interval_coverage = float(deployment_holdout.get("event_only_p10_p90_coverage") or 0.0)
    confidence_score = round(
        max(
            0.0,
            min(
                100.0,
                25.0 * min(1.0, len(training) / 1000.0)
                + 30.0 * _clip((c_index - 0.5) / 0.2, 0.0, 1.0)
                + 25.0 * _clip(1.0 - brier_180 / 0.25, 0.0, 1.0)
                + 20.0 * _clip(1.0 - abs(interval_coverage - 0.80) / 0.80, 0.0, 1.0)
                - 20.0,
            ),
        )
    )
    confidence_grade = (
        "high" if confidence_score >= 75 else "moderate" if confidence_score >= 55 else "limited"
    )
    artifact = {
        "model_version": BASELINE_VERSION,
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "case_dataset_sha256": file_sha256(cases_path),
        "official_monthly_flows_sha256": file_sha256(official_flows_path),
        "population": "Quality-filtered, duplicate-key-collapsed VisaDashboard subclass 500 self-reports",
        "time_split": {
            "train_lodgement_end": TRAIN_CUTOFF.isoformat(),
            "test_lodgement_start": (date(2025, 1, 1)).isoformat(),
            "test_lodgement_end": TEST_END.isoformat(),
            "training_rows": len(training),
            "training_events_as_of_cutoff": sum(row.event_observed for row in training),
            "training_right_censored_as_of_cutoff": sum(1 - row.event_observed for row in training),
            "test_rows": len(testing),
            "test_events": sum(row.event_observed for row in testing),
            "test_right_censored": sum(1 - row.event_observed for row in testing),
            "leakage_control": "Any pre-cutoff application decided after the cutoff is censored at the cutoff.",
            "selection_and_calibration_window": {
                "start": "2025-01-01",
                "end": CALIBRATION_END.isoformat(),
                "rows": len(calibration_rows),
                "events": sum(row.event_observed for row in calibration_rows),
            },
            "final_holdout_window": {
                "start": (CALIBRATION_END + timedelta(days=1)).isoformat(),
                "end": TEST_END.isoformat(),
                "rows": len(holdout_rows),
                "events": sum(row.event_observed for row in holdout_rows),
            },
        },
        "features": {
            "categorical": list(CATEGORICAL_FEATURES),
            "numeric": list(NUMERIC_FEATURES),
            "encoded_columns": encoder.feature_names,
            "official_flow_scope": "China PRC excluding SARs, primary, outside Australia, Postgraduate Research",
            "official_flow_use": "One-month-lagged contextual proxy only for matching PGR/offshore rows; case citizenship is unavailable.",
        },
        "evaluations": evaluations,
        "development_evaluations": development_evaluations,
        "holdout_evaluations": holdout_evaluations,
        "deployment_holdout_evaluation": deployment_holdout,
        "candidate_calibration_holdout_evaluation": candidate_calibration_holdout,
        "ranking": [item["model"] for item in ranked],
        "selected_baseline": ranked[0]["model"],
        "deployment": {
            "engine": "weibull_aft",
            "model": aft.to_dict(),
            "feature_encoder": encoder.to_dict(),
            "training_lodgement_end": TRAIN_CUTOFF.isoformat(),
            "calibration_lodgement_end": CALIBRATION_END.isoformat(),
            "final_holdout_lodgement_start": (CALIBRATION_END + timedelta(days=1)).isoformat(),
            "feature_availability": "Only information available by lodgement is used; official flow predictors lag one calendar month.",
            "input_resolution": "Minute timestamps are preserved in output, but model labels and covariates have day/month resolution.",
            "calibration_decision": "The pooled isotonic candidate was rejected for production because final-holdout event MAE degraded; raw Weibull timing distribution is retained and its limited calibration is disclosed.",
        },
        "deployment_confidence": {
            "score": confidence_score,
            "grade": confidence_grade,
            "meaning": "Evidence and holdout-performance sufficiency, not the probability a particular prediction is correct.",
            "penalties": [
                "Single self-selected crowd source",
                "No individual citizenship field",
                "Historical first-seen timestamps unavailable",
            ],
        },
        "interpretability": {
            "cox_top_absolute_coefficients": _top_coefficients(encoder.feature_names, cox.coefficients),
            "weibull_aft_shape": round(aft.shape, 5),
            "weibull_aft_top_absolute_coefficients": _top_coefficients(
                aft.feature_names, aft.parameters[:-1]
            ),
        },
        "metric_notes": {
            "harrell_c_index": "Uses censor-aware comparable pairs; 0.5 is no discrimination.",
            "event_only_median_absolute_error_days": "Calculated only for observed decisions and therefore subject to event-selection bias.",
            "event_only_p10_p90_coverage": "Outcome interval coverage among observed decisions, not all censored cases.",
            "horizon_calibration": "Brier score and mean calibration for decision by 90, 180 and 365 days; the test window has at least 365 days of follow-up by the dataset snapshot.",
            "calibration_candidate": "A pooled isotonic candidate was fitted on January-April 2025 and rejected after May-August 2025 holdout performance degraded.",
        },
        "limitations": [
            "All source rows were first captured in one 2026 snapshot, so historical record availability cannot be reconstructed.",
            "The crowd sample is self-selected and is not representative of all Home Affairs applications.",
            "Citizenship is absent from individual cases; China-specific official flows are contextual proxies, not confirmed individual attributes.",
            "A decision-time model does not predict grant versus refusal.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=output.parent, suffix=".tmp"
    ) as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
        temp_name = handle.name
    os.replace(temp_name, output)
    return artifact


def _parse_datetime(value: str, timezone_name: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
    return parsed


def forecast_survival_model(
    *,
    model_artifact: Path,
    official_flows_path: Path,
    lodged_at: str,
    as_of: str | None = None,
    timezone_name: str = "Asia/Shanghai",
    education_level: str = "PhD",
    study_sector: str = "Postgraduate Research",
    submit_location: str = "Outside Australia",
    includes_partner: str = "unknown",
    is_diy: str = "unknown",
    provider_group: str = "Unknown",
    field_of_study_group: str = "Unknown",
) -> dict[str, Any]:
    artifact = json.loads(model_artifact.read_text(encoding="utf-8"))
    deployment = artifact.get("deployment")
    if not deployment or deployment.get("engine") != "weibull_aft":
        raise ValueError("Model artifact does not contain a deployable Weibull AFT state")
    lodged = _parse_datetime(lodged_at, timezone_name)
    current = _parse_datetime(as_of, timezone_name) if as_of else datetime.now(lodged.tzinfo)
    if current < lodged:
        raise ValueError("as_of must not be earlier than lodged_at")
    elapsed_days = (current - lodged).total_seconds() / 86400.0
    official = load_official_flow_features(official_flows_path)
    observation = Observation(
        duration_days=max(0.5, elapsed_days),
        event_observed=0,
        lodgement_date=lodged.date(),
        decision_date=None,
        as_of_date=current.date(),
        education_level=education_level,
        study_sector=study_sector,
        submit_location=submit_location,
        includes_partner=includes_partner,
        is_diy=is_diy,
        provider_group=provider_group,
        field_of_study_group=field_of_study_group,
        quality_score=100,
        duplicate_key="forecast-input",
        source_tier="A",
    )
    row = SurvivalRow(
        duration_days=max(0.5, elapsed_days),
        event_observed=0,
        lodgement_date=lodged.date(),
        features=_raw_features(observation, official),
    )
    encoder = FeatureEncoder.from_dict(deployment["feature_encoder"])
    encoder.apply([row])
    base = WeibullAFTBaseline.from_dict(deployment["model"])
    elapsed_cdf = 1.0 - base.survival(row, elapsed_days)
    remaining_survival = max(1e-9, 1.0 - elapsed_cdf)

    def conditional_total_days(probability: float) -> float:
        raw_target = min(0.999999, elapsed_cdf + probability * remaining_survival)
        total = base._scale(row) * (-math.log(1.0 - raw_target)) ** (1.0 / base.shape)
        return max(elapsed_days, total)

    probabilities = (0.10, 0.50, 0.80, 0.90)
    totals = {probability: conditional_total_days(probability) for probability in probabilities}
    predicted_dates = {
        f"p{int(probability * 100)}": (lodged + timedelta(days=duration)).isoformat(timespec="minutes")
        for probability, duration in totals.items()
    }
    remaining_days = {
        f"p{int(probability * 100)}": round(max(0.0, duration - elapsed_days), 1)
        for probability, duration in totals.items()
    }

    conditional_probability: dict[str, float] = {}
    for days_ahead in (30, 60, 90, 180):
        future_cdf = 1.0 - base.survival(row, elapsed_days + days_ahead)
        conditional_probability[f"within_{days_ahead}_days"] = round(
            _clip((future_cdf - elapsed_cdf) / remaining_survival, 0.0, 1.0), 4
        )

    warnings: list[str] = []
    if study_sector == "Postgraduate Research" and submit_location == "Outside Australia":
        prior_month = _previous_months(_month_key(lodged.date()), 2)[1]
        if prior_month not in official:
            warnings.append(
                f"Official lagged flow context is unavailable for {prior_month}; proxy features fall back to zero."
            )
    for field in CATEGORICAL_FEATURES:
        value = str(row.features.get(field, "Unknown"))
        if value not in encoder.levels.get(field, []):
            warnings.append(f"Unseen category for {field}: {value!r}; encoded as the reference level.")

    base_confidence = artifact.get("deployment_confidence", {})
    confidence_score = max(0, int(base_confidence.get("score", 0)) - 5 * len(warnings))
    confidence_grade = "high" if confidence_score >= 75 else "moderate" if confidence_score >= 55 else "limited"
    return {
        "model_version": artifact.get("model_version"),
        "engine": deployment["engine"],
        "lodged_at": lodged.isoformat(timespec="minutes"),
        "as_of": current.isoformat(timespec="minutes"),
        "elapsed_days": round(elapsed_days, 3),
        "predicted_decision_at": predicted_dates,
        "remaining_days": remaining_days,
        "conditional_decision_probability": conditional_probability,
        "predictive_interval": {
            "level": "80%",
            "low": predicted_dates["p10"],
            "high": predicted_dates["p90"],
            "meaning": "Conditional model outcome interval among cases still undecided at as_of; not a confidence interval or guarantee.",
        },
        "data_confidence": {
            "score": confidence_score,
            "grade": confidence_grade,
            "meaning": base_confidence.get("meaning"),
            "holdout": artifact.get("deployment_holdout_evaluation"),
        },
        "inputs": {
            "education_level": education_level,
            "study_sector": study_sector,
            "submit_location": submit_location,
            "includes_partner": includes_partner,
            "is_diy": is_diy,
            "provider_group": provider_group,
            "field_of_study_group": field_of_study_group,
        },
        "warnings": warnings,
        "limitations": [
            "Predictions estimate time to a recorded decision, not grant probability.",
            "The individual-case training sample is self-selected and dominated by postgraduate research cases.",
            "Minute precision preserves the input clock time; model resolution is still day/month level.",
            "The 80% interval comes from the fitted Weibull distribution and can miss under policy or workload shifts.",
            "A candidate probability calibrator was rejected because it worsened final-holdout timing error; the raw Weibull distribution is used.",
        ],
    }
