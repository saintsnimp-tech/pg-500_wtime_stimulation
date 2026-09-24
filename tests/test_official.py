from __future__ import annotations

import unittest

from visa_wait.official import _calendar_month, validate_monthly_flows


class OfficialMonthlyFlowTests(unittest.TestCase):
    def test_financial_year_month_mapping(self) -> None:
        self.assertEqual(_calendar_month("2025-26", "M01 Jul"), "2025-07")
        self.assertEqual(_calendar_month("2025-26", "M06 Dec"), "2025-12")
        self.assertEqual(_calendar_month("2025-26", "M07 Jan"), "2026-01")
        self.assertEqual(_calendar_month("2025-26", "M12 Jun"), "2026-06")

    def test_expected_month_activity_grain_is_valid(self) -> None:
        rows = []
        year, month = 2015, 1
        while (year, month) <= (2026, 8):
            key = f"{year:04d}-{month:02d}"
            rows.extend(
                {
                    "source_snapshot_date": "2026-08-31",
                    "activity_month": key,
                    "activity_type": activity,
                    "count": 1,
                }
                for activity in ("lodged", "granted")
            )
            month += 1
            if month == 13:
                year, month = year + 1, 1
        result = validate_monthly_flows(rows)
        self.assertEqual(result["rows"], 280)
        self.assertEqual(result["issues"], [])


if __name__ == "__main__":
    unittest.main()
