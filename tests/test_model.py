from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from visa_wait.baselines import forecast_survival_model, prepare_time_split
from visa_wait.model import (
    Observation,
    WeightedPoint,
    curve_quantile,
    forecast,
    weighted_km_curve,
)


class WeightedSurvivalTests(unittest.TestCase):
    def test_weighted_km_quantiles_are_ordered(self) -> None:
        curve = weighted_km_curve(
            [WeightedPoint(5, 1, 1), WeightedPoint(10, 0, 1), WeightedPoint(20, 1, 1), WeightedPoint(30, 1, 1)]
        )
        self.assertEqual(curve_quantile(curve, 0.10), 5)
        self.assertEqual(curve_quantile(curve, 0.50), 20)
        self.assertEqual(curve_quantile(curve, 0.90), 30)

    def test_forecast_preserves_input_minute(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = forecast(
            cases_path=root / "data" / "11_visadashboard_cases.csv",
            official_path=root / "data" / "01_official_current_streams_2026-09.csv",
            lodged_at="2026-09-24T15:37+08:00",
            as_of="2026-09-24T15:37+08:00",
            education_level="PhD",
            study_sector="Postgraduate Research",
            submit_location="Outside Australia",
            model_artifact=root / "models" / "subclass500_survival_v3.json",
        )
        self.assertRegex(result["predicted_decision_at"]["p50"], r"T\d{2}:\d{2}\+08:00$")
        self.assertLess(result["remaining_days"]["p50"], result["remaining_days"]["p90"])
        self.assertEqual(result["backtest"]["selected_baseline"], "weibull_aft")

    def test_time_split_censors_decisions_after_training_cutoff(self) -> None:
        observation = Observation(
            duration_days=410,
            event_observed=1,
            lodgement_date=date(2024, 7, 1),
            decision_date=date(2025, 8, 15),
            as_of_date=date(2026, 9, 24),
            education_level="PhD",
            study_sector="Postgraduate Research",
            submit_location="Outside Australia",
            includes_partner="unknown",
            is_diy="unknown",
            provider_group="Unknown",
            field_of_study_group="Unknown",
            quality_score=80,
            duplicate_key="example",
            source_tier="B-C",
        )
        training, testing = prepare_time_split([observation], {})
        self.assertEqual(len(testing), 0)
        self.assertEqual(training[0].event_observed, 0)
        self.assertEqual(training[0].duration_days, 183.0)

    def test_deployed_forecast_is_ordered_and_conditional(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = forecast_survival_model(
            model_artifact=root / "models" / "subclass500_survival_v3.json",
            official_flows_path=root / "data" / "14_official_monthly_flows_2015-2026.csv",
            lodged_at="2026-08-20T10:17+08:00",
            as_of="2026-09-24T10:17+08:00",
            education_level="PhD",
            study_sector="Postgraduate Research",
            submit_location="Outside Australia",
        )
        self.assertEqual(result["engine"], "weibull_aft")
        self.assertLessEqual(result["remaining_days"]["p10"], result["remaining_days"]["p50"])
        self.assertLessEqual(result["remaining_days"]["p50"], result["remaining_days"]["p90"])
        self.assertEqual(result["data_confidence"]["grade"], "limited")


if __name__ == "__main__":
    unittest.main()
