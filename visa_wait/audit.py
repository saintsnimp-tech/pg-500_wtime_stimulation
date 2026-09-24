from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


PREDICTOR_FIELDS = [
    "education_level",
    "study_sector",
    "field_of_study_group",
    "submit_location",
    "includes_partner",
    "is_diy",
    "provider_group",
]


def _date(value: str) -> date | None:
    try:
        return date.fromisoformat((value or "")[:10])
    except ValueError:
        return None


def _percentage(value: int, total: int) -> float:
    return round(value / total, 4) if total else 0.0


def audit_case_data(
    path: Path,
    *,
    quality_threshold: int = 40,
    train_cutoff: date = date(2024, 12, 31),
    test_end: date = date(2025, 8, 31),
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)

    record_ids: Counter[str] = Counter(row.get("record_id", "") for row in rows)
    duplicate_keys: Counter[str] = Counter(
        row.get("duplicate_key", "") for row in rows if row.get("duplicate_key", "")
    )
    exact_rows: Counter[tuple[tuple[str, str], ...]] = Counter(
        tuple(sorted(row.items())) for row in rows
    )
    issue_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    lodgement_years: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    valid_candidates: list[dict[str, str]] = []

    for row in rows:
        status_counts[row.get("application_status", "(blank)")] += 1
        source_counts[row.get("source_id", "(blank)")] += 1
        lodged = _date(row.get("lodgement_date", ""))
        decision = _date(row.get("decision_date", ""))
        as_of = _date(row.get("as_of_date", ""))
        snapshot = _date(row.get("snapshot_date", ""))
        if lodged:
            lodgement_years[str(lodged.year)] += 1
        else:
            issue_counts["missing_or_invalid_lodgement_date"] += 1
        try:
            event = int(row.get("event_observed", ""))
            waiting = int(row.get("waiting_days", ""))
            quality = int(row.get("record_quality_score", ""))
        except ValueError:
            issue_counts["invalid_numeric_field"] += 1
            continue

        quality_counts["below_threshold" if quality < quality_threshold else "usable_quality"] += 1
        if event not in (0, 1):
            issue_counts["invalid_event_observed"] += 1
        if waiting < 0 or waiting > 3650:
            issue_counts["implausible_waiting_days"] += 1
        if lodged and snapshot and lodged > snapshot:
            issue_counts["future_lodgement"] += 1
        if decision and lodged and decision < lodged:
            issue_counts["decision_before_lodgement"] += 1
        if event == 1 and decision is None:
            issue_counts["event_without_decision_date"] += 1
        if event == 0 and decision is not None:
            issue_counts["censored_with_decision_date"] += 1
        if lodged and as_of and waiting != (as_of - lodged).days:
            issue_counts["waiting_days_mismatch"] += 1
        expected_status = "granted" if event == 1 else "pending"
        if row.get("application_status") != expected_status:
            issue_counts["status_event_mismatch"] += 1
        for field in PREDICTOR_FIELDS:
            if str(row.get(field, "")).strip().lower() in {"", "unknown", "(blank)"}:
                unknown_counts[field] += 1
        if quality >= quality_threshold and lodged and as_of and 0 <= waiting <= 3650 and event in (0, 1):
            valid_candidates.append(row)

    duplicate_record_ids = sum(1 for key, count in record_ids.items() if key and count > 1)
    duplicate_record_rows = sum(count for key, count in record_ids.items() if key and count > 1)
    duplicate_timeline_groups = sum(1 for count in duplicate_keys.values() if count > 1)
    duplicate_timeline_rows = sum(count for count in duplicate_keys.values() if count > 1)
    exact_duplicate_rows = sum(count - 1 for count in exact_rows.values() if count > 1)

    deduplicated: dict[str, dict[str, str]] = {}
    for row in valid_candidates:
        key = row.get("duplicate_key") or row.get("record_id", "")
        prior = deduplicated.get(key)
        if prior is None or int(row["record_quality_score"]) > int(prior["record_quality_score"]):
            deduplicated[key] = row
    model_rows = list(deduplicated.values())
    train_rows = [row for row in model_rows if (_date(row["lodgement_date"]) or date.max) <= train_cutoff]
    test_rows = [
        row
        for row in model_rows
        if train_cutoff < (_date(row["lodgement_date"]) or date.min) <= test_end
    ]

    findings: list[dict[str, Any]] = []
    invalid_date_total = sum(
        issue_counts[name]
        for name in (
            "missing_or_invalid_lodgement_date",
            "future_lodgement",
            "decision_before_lodgement",
            "waiting_days_mismatch",
        )
    )
    if duplicate_record_ids or invalid_date_total:
        findings.append(
            {
                "severity": "critical",
                "finding": "Primary-key or timeline validity failures",
                "evidence": {
                    "duplicate_record_id_groups": duplicate_record_ids,
                    "invalid_date_or_duration_rows": invalid_date_total,
                },
                "impact": "Can duplicate or corrupt survival durations.",
            }
        )
    findings.append(
        {
            "severity": "high" if duplicate_timeline_groups else "low",
            "finding": "Potential duplicate timelines",
            "evidence": {
                "groups": duplicate_timeline_groups,
                "affected_rows": duplicate_timeline_rows,
                "rows_after_quality_and_dedup": len(model_rows),
            },
            "impact": "Uncollapsed timelines overweight repeated self-reports; training uses one highest-quality row per key.",
        }
    )
    findings.append(
        {
            "severity": "high",
            "finding": "Citizenship is unavailable in the individual-case table",
            "evidence": {"citizenship_column_present": "citizenship_country" in fields},
            "impact": "The crowd model cannot claim a China-only individual population; official China flows are contextual proxies only.",
        }
    )
    findings.append(
        {
            "severity": "high",
            "finding": "Source availability time is unavailable",
            "evidence": {"source_ids": dict(source_counts), "snapshot_dates": sorted({row.get("snapshot_date", "") for row in rows})},
            "impact": "Outcome-time leakage can be prevented, but a perfect historical replay of when crowd records first appeared is impossible.",
        }
    )

    return {
        "dataset": str(path),
        "grain": "One anonymized self-reported application timeline per source record before duplicate-key collapse",
        "rows": len(rows),
        "columns": len(fields),
        "column_names": fields,
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "lodgement_year_counts": dict(sorted(lodgement_years.items())),
        "quality": {
            **dict(quality_counts),
            "quality_threshold": quality_threshold,
            "usable_rate": _percentage(quality_counts["usable_quality"], len(rows)),
        },
        "uniqueness": {
            "duplicate_record_id_groups": duplicate_record_ids,
            "duplicate_record_id_rows": duplicate_record_rows,
            "exact_duplicate_rows_beyond_first": exact_duplicate_rows,
            "duplicate_timeline_groups": duplicate_timeline_groups,
            "duplicate_timeline_rows": duplicate_timeline_rows,
            "rows_after_quality_and_dedup": len(model_rows),
            "collapsed_rows": len(valid_candidates) - len(model_rows),
        },
        "validity_issue_counts": dict(issue_counts),
        "predictor_unknown_rates": {
            field: _percentage(unknown_counts[field], len(rows)) for field in PREDICTOR_FIELDS
        },
        "temporal_split": {
            "train_cutoff": train_cutoff.isoformat(),
            "test_end": test_end.isoformat(),
            "training_rows_before_as_of_reconstruction": len(train_rows),
            "test_rows": len(test_rows),
            "test_observed_events": sum(int(row["event_observed"]) for row in test_rows),
            "test_right_censored": sum(1 - int(row["event_observed"]) for row in test_rows),
        },
        "findings": findings,
        "training_policy": {
            "quality_threshold": quality_threshold,
            "duplicate_policy": "Keep the highest-quality row per duplicate_key",
            "censoring_policy": "Pending rows are right-censored at as_of_date",
            "time_split_policy": "Training outcomes after the cutoff are censored at the cutoff; test cohorts lodge after the cutoff.",
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VisaDashboard model-data audit",
        "",
        f"Dataset: `{report['dataset']}`",
        "",
        "## Summary",
        "",
        f"- Raw rows: {report['rows']}",
        f"- Rows after quality filtering and duplicate-key collapse: {report['uniqueness']['rows_after_quality_and_dedup']}",
        f"- Potential duplicate timeline groups: {report['uniqueness']['duplicate_timeline_groups']}",
        f"- Test window: {report['temporal_split']['train_cutoff']} to {report['temporal_split']['test_end']}",
        f"- Test rows/events/censored: {report['temporal_split']['test_rows']} / {report['temporal_split']['test_observed_events']} / {report['temporal_split']['test_right_censored']}",
        "",
        "## Findings",
        "",
    ]
    for finding in report["findings"]:
        lines.extend(
            [
                f"### {str(finding['severity']).upper()}: {finding['finding']}",
                "",
                finding["impact"],
                "",
                "Evidence: `" + json.dumps(finding["evidence"], ensure_ascii=False, sort_keys=True) + "`",
                "",
            ]
        )
    lines.extend(
        [
            "## Modeling controls",
            "",
            "- Do not use decision-date fields as predictors.",
            "- Reconstruct training censoring at the cutoff date.",
            "- Keep official monthly flow values as contextual proxy features because individual citizenship is absent.",
            "- Report event-only MAE separately from censored-data discrimination and horizon calibration.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_report(input_path: Path, json_output: Path, markdown_output: Path) -> dict[str, Any]:
    report = audit_case_data(input_path)
    for output, content in (
        (json_output, json.dumps(report, ensure_ascii=False, indent=2)),
        (markdown_output, _markdown(report)),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=output.parent, suffix=".tmp"
        ) as handle:
            handle.write(content)
            temp_name = handle.name
        os.replace(temp_name, output)
    return report
