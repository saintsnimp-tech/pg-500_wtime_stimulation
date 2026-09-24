from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable

from .analysis import estimate_survival
from .http import HttpPolicy, PoliteJsonClient
from .schema import NormalizedCase, SCHEMA_FIELDS
from .sources import collect_visadashboard


def _atomic_csv_write(path: Path, rows: Iterable[NormalizedCase]) -> tuple[int, Counter[str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    statuses: Counter[str] = Counter()
    seen: set[str] = set()
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA_FIELDS)
        writer.writeheader()
        for case in rows:
            if case.record_id in seen:
                continue
            seen.add(case.record_id)
            writer.writerow(case.as_row())
            count += 1
            statuses[case.application_status] += 1
        temp_name = handle.name
    os.replace(temp_name, path)
    return count, statuses


def collect_visadashboard_to_csv(
    output: Path,
    snapshot_date: date,
    page_size: int = 100,
    max_pages: int | None = None,
    respect_robots: bool = True,
) -> dict[str, object]:
    client = PoliteJsonClient(HttpPolicy(respect_robots=respect_robots))
    rows = collect_visadashboard(client, snapshot_date, page_size=page_size, max_pages=max_pages)
    count, statuses = _atomic_csv_write(output, rows)
    return {"output": str(output), "rows": count, "statuses": dict(statuses)}


def validate_case_csv(path: Path) -> dict[str, object]:
    issues: list[str] = []
    record_ids: set[str] = set()
    rows = 0
    usable = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in SCHEMA_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            issues.append("missing_columns:" + ",".join(missing))
        for line_number, row in enumerate(reader, start=2):
            rows += 1
            record_id = row.get("record_id", "")
            if not record_id:
                issues.append(f"line_{line_number}:missing_record_id")
            elif record_id in record_ids:
                issues.append(f"line_{line_number}:duplicate_record_id")
            record_ids.add(record_id)
            try:
                waiting = int(row.get("waiting_days", ""))
                event = int(row.get("event_observed", ""))
                score = int(row.get("record_quality_score", ""))
                if 0 <= waiting <= 3650 and event in (0, 1) and score >= 40:
                    usable += 1
            except ValueError:
                pass
    return {"path": str(path), "rows": rows, "usable_for_survival": usable, "issues": issues}


def profile_case_csv(path: Path, output: Path) -> dict[str, object]:
    dimensions = [
        "application_status",
        "education_level",
        "study_sector",
        "submit_location",
        "includes_partner",
        "is_diy",
        "provider_group",
        "field_of_study_group",
    ]
    counts: dict[str, Counter[str]] = {name: Counter() for name in dimensions}
    counts["quality_flag"] = Counter()
    snapshot_date = ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            snapshot_date = snapshot_date or row.get("snapshot_date", "")
            for name in dimensions:
                counts[name][row.get(name, "") or "(blank)"] += 1
            for flag in filter(None, row.get("quality_flags", "").split("|")):
                counts["quality_flag"][flag] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=output.parent, suffix=".tmp"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dimension", "value", "row_count", "snapshot_date", "notes"]
        )
        writer.writeheader()
        row_count = 0
        for dimension, counter in counts.items():
            for value, count in counter.most_common():
                writer.writerow(
                    {
                        "dimension": dimension,
                        "value": value,
                        "row_count": count,
                        "snapshot_date": snapshot_date,
                        "notes": "Derived from anonymized normalized case data",
                    }
                )
                row_count += 1
        temp_name = handle.name
    os.replace(temp_name, output)
    return {"input": str(path), "output": str(output), "rows": row_count}


def predict_from_csv(
    path: Path,
    education_level: str | None = None,
    submit_location: str | None = None,
) -> dict[str, object]:
    observations: list[tuple[int, int]] = []
    seen_duplicate_keys: set[str] = set()
    collapsed_duplicates = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if education_level and row.get("education_level") != education_level:
                continue
            if submit_location and row.get("submit_location") != submit_location:
                continue
            try:
                if int(row.get("record_quality_score", "0")) < 40:
                    continue
                duplicate_key = row.get("duplicate_key", "")
                if duplicate_key and duplicate_key in seen_duplicate_keys:
                    collapsed_duplicates += 1
                    continue
                if duplicate_key:
                    seen_duplicate_keys.add(duplicate_key)
                observations.append((int(row["waiting_days"]), int(row["event_observed"])))
            except (KeyError, ValueError):
                continue
    result = estimate_survival(observations)
    return {
        **result.__dict__,
        "collapsed_potential_duplicates": collapsed_duplicates,
        "filters": {"education_level": education_level, "submit_location": submit_location},
        "method": "Kaplan-Meier with right-censored pending cases; bootstrap 95% CI for median",
        "warning": "Crowdsourced historical estimate, not an official decision deadline or migration advice.",
    }


def print_json(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
