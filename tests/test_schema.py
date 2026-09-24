from __future__ import annotations

import unittest
from datetime import date

from visa_wait.analysis import kaplan_meier_quantiles
from visa_wait.schema import normalize_visadashboard_record


class SchemaTests(unittest.TestCase):
    def test_granted_case_is_normalized_and_free_text_is_not_retained(self) -> None:
        raw = {
            "_id": "abc123",
            "educationLevel": "博士",
            "submitTime": "2026-01-01",
            "getVisaTime": "2026-01-21",
            "ifGetVisa": "true",
            "ifDIY": "true",
            "ifIncludedCouple": "单独学签",
            "submitPlace": "国内递交",
            "major": "计算机科学",
            "schoolType": "Monash",
            "visaOfficer": "secret-officer-code",
            "otherInfo": "free text that must be dropped",
        }
        case = normalize_visadashboard_record(raw, date(2026, 9, 24), "https://example.test")
        self.assertEqual(case.waiting_days, 20)
        self.assertEqual(case.event_observed, 1)
        self.assertEqual(case.education_level, "PhD")
        self.assertEqual(case.field_of_study_group, "Computing and Data")
        self.assertNotIn("secret-officer-code", repr(case))
        self.assertNotIn("free text", repr(case))

    def test_pending_case_is_right_censored_at_snapshot(self) -> None:
        case = normalize_visadashboard_record(
            {"_id": "pending", "submitTime": "2026-09-01", "ifGetVisa": "false"},
            date(2026, 9, 24),
            "https://example.test",
        )
        self.assertEqual(case.waiting_days, 23)
        self.assertEqual(case.event_observed, 0)
        self.assertEqual(case.application_status, "pending")


class SurvivalTests(unittest.TestCase):
    def test_kaplan_meier_keeps_censored_cases_in_risk_set(self) -> None:
        p10, median, p90 = kaplan_meier_quantiles([(5, 1), (8, 0), (10, 1), (20, 1)])
        self.assertEqual(p10, 5)
        self.assertEqual(median, 10)
        self.assertEqual(p90, 20)


if __name__ == "__main__":
    unittest.main()

