from __future__ import annotations

import unittest
from pathlib import Path

from visa_wait.audit import audit_case_data


class ModelingDataAuditTests(unittest.TestCase):
    def test_current_dataset_audit_counts_are_stable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = audit_case_data(root / "data" / "11_visadashboard_cases.csv")
        self.assertEqual(report["rows"], 1511)
        self.assertEqual(report["uniqueness"]["duplicate_record_id_groups"], 0)
        self.assertEqual(report["uniqueness"]["rows_after_quality_and_dedup"], 1460)
        self.assertEqual(report["temporal_split"]["test_rows"], 317)
        self.assertFalse("citizenship_country" in report["column_names"])


if __name__ == "__main__":
    unittest.main()
