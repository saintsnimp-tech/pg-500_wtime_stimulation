from __future__ import annotations

import csv
import html
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable

from .model import (
    curve_quantile,
    load_observations,
    load_official_anchors,
    weighted_km_curve,
    weighted_points,
)


BLUE = "#2563EB"
GOLD = "#D4A72C"
ORANGE = "#D97706"
CHARCOAL = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
BACKGROUND = "#FFFFFF"


def _svg_text(x: float, y: float, value: object, size: int = 15, anchor: str = "start", color: str = CHARCOAL, weight: int = 400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Segoe UI,Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">'
        f"{html.escape(str(value))}</text>"
    )


def _write_svg(path: Path, body: Iterable[str], width: int = 1200, height: int = 700) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(document), encoding="utf-8")


def render_survival_evidence(cases_path: Path, official_path: Path, output: Path) -> dict[str, object]:
    observations = load_observations(cases_path)
    reference = max((item.as_of_date for item in observations), default=date.today())
    overall = weighted_km_curve(weighted_points(observations, reference))
    phd_offshore = [
        item for item in observations
        if item.education_level == "PhD" and item.submit_location == "Outside Australia"
    ]
    cohort = weighted_km_curve(weighted_points(phd_offshore, reference))
    anchor = load_official_anchors(official_path).get("Postgraduate Research Sector")

    width, height = 1200, 700
    left, right, top, bottom = 95, 55, 95, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    max_x = min(900.0, max((point[0] for point in overall), default=365.0))

    def xy(duration: float, survival: float) -> tuple[float, float]:
        return left + duration / max_x * plot_w, top + (1.0 - survival) * plot_h

    body: list[str] = [
        _svg_text(left, 38, "Subclass 500 crowd survival evidence", 25, weight=650),
        _svg_text(left, 66, f"Weighted Kaplan–Meier; snapshot {reference.isoformat()}; pending cases are right-censored", 14, color=MUTED),
    ]
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + fraction * plot_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{GRID}"/>')
        body.append(_svg_text(left - 12, y + 5, f"{int((1-fraction)*100)}%", 13, "end", MUTED))
    for day in range(0, int(max_x) + 1, 100):
        x, _ = xy(day, 0)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{GRID}"/>')
        body.append(_svg_text(x, top + plot_h + 28, day, 13, "middle", MUTED))

    for curve, color, label in ((overall, CHARCOAL, "All usable crowd cases"), (cohort, BLUE, "PhD · Outside Australia")):
        clipped = [(d, s) for d, s in curve if d <= max_x]
        points = " ".join(f"{xy(d, s)[0]:.1f},{xy(d, s)[1]:.1f}" for d, s in clipped)
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        median = curve_quantile(curve, 0.5)
        if median is not None and median <= max_x:
            x, y = xy(median, 0.5)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
            body.append(_svg_text(x + 8, y - 8, f"{label} median {median:.0f}d", 13, color=color, weight=600))

    if anchor:
        for value, label, dash in ((anchor.p50_days, "Official P50", "7 5"), (anchor.p90_days, "Official P90", "2 5")):
            x, _ = xy(value, 0)
            body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{GOLD}" stroke-width="3" stroke-dasharray="{dash}"/>')
            body.append(_svg_text(x + 6, top + 20, f"{label} {value:.0f}d", 13, color=ORANGE, weight=600))

    body.extend([
        _svg_text(left + plot_w / 2, height - 24, "Observed total processing time (calendar days)", 15, "middle", CHARCOAL, 600),
        _svg_text(20, top + plot_h / 2, "Share not yet decided", 14, "start", CHARCOAL, 600),
        _svg_text(left, height - 52, "Official markers describe recently decided applications; crowd curves are self-selected historical records.", 13, color=MUTED),
    ])
    _write_svg(output, body, width, height)
    return {"output": str(output), "overall_rows": len(observations), "cohort_rows": len(phd_offshore)}


def render_monthly_volume(cases_path: Path, output: Path) -> dict[str, object]:
    lodgements: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    with cases_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("lodgement_date"):
                lodgements[row["lodgement_date"][:7]] += 1
            if row.get("decision_date"):
                decisions[row["decision_date"][:7]] += 1
    months = sorted(set(lodgements) | set(decisions))[-30:]
    width, height = 1200, 700
    left, right, top, bottom = 95, 55, 95, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    max_y = max([lodgements[m] for m in months] + [decisions[m] for m in months] + [1])
    body: list[str] = [
        _svg_text(left, 38, "Monthly source-case volume", 25, weight=650),
        _svg_text(left, 66, "Counts reflect crowd-source coverage, not Australian visa application volume", 14, color=MUTED),
    ]
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1 - fraction) * plot_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{GRID}"/>')
        body.append(_svg_text(left - 12, y + 5, round(max_y * fraction), 13, "end", MUTED))
    step = plot_w / max(1, len(months))
    bar_w = max(5.0, step * 0.32)
    for index, month in enumerate(months):
        center = left + (index + 0.5) * step
        lodgement_h = lodgements[month] / max_y * plot_h
        decision_h = decisions[month] / max_y * plot_h
        body.append(f'<rect x="{center - bar_w - 1:.1f}" y="{top + plot_h - lodgement_h:.1f}" width="{bar_w:.1f}" height="{lodgement_h:.1f}" fill="{BLUE}"/>')
        body.append(f'<rect x="{center + 1:.1f}" y="{top + plot_h - decision_h:.1f}" width="{bar_w:.1f}" height="{decision_h:.1f}" fill="{GOLD}"/>')
        if index % 3 == 0 or index == len(months) - 1:
            body.append(_svg_text(center, top + plot_h + 28, month, 12, "middle", MUTED))
    body.extend([
        f'<rect x="{left}" y="{height - 56}" width="14" height="14" fill="{BLUE}"/>',
        _svg_text(left + 22, height - 44, "Lodged", 13),
        f'<rect x="{left + 110}" y="{height - 56}" width="14" height="14" fill="{GOLD}"/>',
        _svg_text(left + 132, height - 44, "Decision observed", 13),
    ])
    _write_svg(output, body, width, height)
    return {"output": str(output), "months": len(months)}


def render_backtest(model_path: Path, output: Path) -> dict[str, object]:
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    evaluations = artifact.get("holdout_evaluations") or artifact.get("evaluations") or []
    width, height = 1200, 690
    left, top, plot_w, plot_h = 250, 130, 850, 400
    body: list[str] = [
        _svg_text(70, 42, "Final-holdout survival baseline comparison", 25, weight=650),
        _svg_text(
            70,
            72,
            f"Train through {artifact.get('time_split', {}).get('train_lodgement_end', 'n/a')} · "
            f"final holdout {artifact.get('time_split', {}).get('final_holdout_window', {}).get('start', 'n/a')} to "
            f"{artifact.get('time_split', {}).get('final_holdout_window', {}).get('end', 'n/a')}",
            14,
            color=MUTED,
        ),
    ]
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + fraction * plot_w
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{GRID}"/>')
        body.append(_svg_text(x, top + plot_h + 28, f"{fraction:.2f}", 13, "middle", MUTED))
    row_height = plot_h / max(1, len(evaluations))
    for index, result in enumerate(evaluations):
        y = top + index * row_height + row_height * 0.18
        c_index = float(result.get("harrell_c_index") or 0)
        brier = float(result.get("horizon_calibration", {}).get("180", {}).get("brier_score") or 0)
        body.append(_svg_text(left - 14, y + 21, result.get("model", "unknown"), 14, "end", CHARCOAL, 600))
        body.append(f'<rect x="{left}" y="{y:.1f}" width="{plot_w * c_index:.1f}" height="22" fill="{BLUE}"/>')
        body.append(f'<rect x="{left}" y="{y + 28:.1f}" width="{plot_w * brier:.1f}" height="15" fill="{GOLD}"/>')
        body.append(_svg_text(left + plot_w * c_index + 8, y + 18, f"C {c_index:.3f}", 13, color=BLUE, weight=650))
        body.append(_svg_text(left + plot_w * brier + 8, y + 41, f"Brier180 {brier:.3f}", 12, color=ORANGE, weight=600))
    body.append(_svg_text(70, 610, f"Deployed engine: {artifact.get('deployment', {}).get('engine', 'n/a')} · confidence {artifact.get('deployment_confidence', {}).get('grade', 'n/a')}", 16, weight=650))
    body.append(_svg_text(70, 642, "C-index: higher is better. Brier score: lower is better. Crowd-source historical availability remains unreconstructable.", 14, color=MUTED))
    _write_svg(output, body, width, height)
    return {"output": str(output), "models": len(evaluations), "selected": artifact.get("selected_baseline")}


def render_all(cases_path: Path, official_path: Path, model_path: Path, output_dir: Path) -> dict[str, object]:
    return {
        "survival": render_survival_evidence(cases_path, official_path, output_dir / "survival_evidence.svg"),
        "volume": render_monthly_volume(cases_path, output_dir / "monthly_source_volume.svg"),
        "backtest": render_backtest(model_path, output_dir / "backtest_calibration.svg"),
    }
