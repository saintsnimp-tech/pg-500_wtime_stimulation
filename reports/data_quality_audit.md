# VisaDashboard model-data audit

Dataset: `data\11_visadashboard_cases.csv`

## Summary

- Raw rows: 1511
- Rows after quality filtering and duplicate-key collapse: 1460
- Potential duplicate timeline groups: 41
- Test window: 2024-12-31 to 2025-08-31
- Test rows/events/censored: 317 / 270 / 47

## Findings

### CRITICAL: Primary-key or timeline validity failures

Can duplicate or corrupt survival durations.

Evidence: `{"duplicate_record_id_groups": 0, "invalid_date_or_duration_rows": 2}`

### HIGH: Potential duplicate timelines

Uncollapsed timelines overweight repeated self-reports; training uses one highest-quality row per key.

Evidence: `{"affected_rows": 90, "groups": 41, "rows_after_quality_and_dedup": 1460}`

### HIGH: Citizenship is unavailable in the individual-case table

The crowd model cannot claim a China-only individual population; official China flows are contextual proxies only.

Evidence: `{"citizenship_column_present": false}`

### HIGH: Source availability time is unavailable

Outcome-time leakage can be prevented, but a perfect historical replay of when crowd records first appeared is impossible.

Evidence: `{"snapshot_dates": ["2026-09-24"], "source_ids": {"PLAT-VISADASHBOARD": 1511}}`

## Modeling controls

- Do not use decision-date fields as predictors.
- Reconstruct training censoring at the cutoff date.
- Keep official monthly flow values as contextual proxy features because individual citizenship is absent.
- Report event-only MAE separately from censored-data discrimination and horizon calibration.
