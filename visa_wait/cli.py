from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .pipeline import (
    collect_visadashboard_to_csv,
    predict_from_csv,
    print_json,
    profile_case_csv,
    validate_case_csv,
)
from .audit import write_audit_report
from .baselines import forecast_survival_model, train_survival_baselines
from .model import forecast
from .official import PACKAGE_URL, ResourceInfo, refresh_official_monthly_flows
from .social import collect_public_urls, collect_x_recent, write_social_csv
from .visualize import render_all


DEFAULT_DATASET = Path("data/11_visadashboard_cases.csv")
DEFAULT_PROFILE = Path("data/12_visadashboard_profile_2026-09.csv")
DEFAULT_SOCIAL = Path("data/13_social_evidence.csv")
DEFAULT_OFFICIAL = Path("data/01_official_current_streams_2026-09.csv")
DEFAULT_OFFICIAL_MONTHLY = Path("data/14_official_monthly_flows_2015-2026.csv")
DEFAULT_OFFICIAL_REVISIONS = Path("data/15_official_monthly_flow_revisions.csv")
DEFAULT_OFFICIAL_SNAPSHOTS = Path("data/snapshots")
DEFAULT_AUDIT_JSON = Path("reports/data_quality_audit.json")
DEFAULT_AUDIT_MD = Path("reports/data_quality_audit.md")
DEFAULT_MODEL = Path("models/subclass500_survival_v3.json")
DEFAULT_FIGURES = Path("reports/figures")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visa-wait",
        description="Collect and validate public Australian subclass 500 waiting-time data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Collect a public source into the canonical schema")
    collect.add_argument("source", choices=["visadashboard"])
    collect.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    collect.add_argument("--snapshot-date", type=date.fromisoformat, default=date.today())
    collect.add_argument("--page-size", type=int, default=100)
    collect.add_argument("--max-pages", type=int)
    collect.add_argument("--ignore-robots", action="store_true", help="Use only after a manual policy review")

    validate = sub.add_parser("validate", help="Validate the canonical case dataset")
    validate.add_argument("--input", type=Path, default=DEFAULT_DATASET)

    profile = sub.add_parser("profile", help="Rebuild a categorized audit profile")
    profile.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    profile.add_argument("--output", type=Path, default=DEFAULT_PROFILE)

    predict = sub.add_parser("predict", help="Produce a censored empirical waiting-time baseline")
    predict.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    predict.add_argument("--education-level", choices=["PhD", "Master", "Bachelor", "School", "ELICOS", "VET"])
    predict.add_argument("--submit-location", choices=["In Australia", "Outside Australia"])

    social = sub.add_parser("social-collect", help="Collect permitted public social evidence")
    social.add_argument("source", choices=["urls", "x"])
    social.add_argument("--input", type=Path, default=Path("config/social_public_urls.txt"))
    social.add_argument("--output", type=Path, default=DEFAULT_SOCIAL)
    social.add_argument(
        "--query",
        default='("subclass 500" OR "澳洲学签") (lodged OR granted OR 递签 OR 下签) -is:retweet',
    )
    social.add_argument("--max-pages", type=int, default=3)
    social.add_argument("--ignore-robots", action="store_true", help="Use only after a manual policy review")

    official = sub.add_parser(
        "official-refresh",
        help="Refresh the exact-filter BP0015 official monthly lodged/granted series",
    )
    official.add_argument("--output", type=Path, default=DEFAULT_OFFICIAL_MONTHLY)
    official.add_argument("--snapshots-dir", type=Path, default=DEFAULT_OFFICIAL_SNAPSHOTS)
    official.add_argument("--revisions", type=Path, default=DEFAULT_OFFICIAL_REVISIONS)
    official.add_argument("--lodged-xlsx", type=Path)
    official.add_argument("--granted-xlsx", type=Path)

    audit = sub.add_parser("audit", help="Audit modeling readiness and leakage-safe split counts")
    audit.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    audit.add_argument("--json-output", type=Path, default=DEFAULT_AUDIT_JSON)
    audit.add_argument("--markdown-output", type=Path, default=DEFAULT_AUDIT_MD)

    train = sub.add_parser("train", help="Train and time-split-test survival model baselines")
    train.add_argument("--cases", type=Path, default=DEFAULT_DATASET)
    train.add_argument("--official-flows", type=Path, default=DEFAULT_OFFICIAL_MONTHLY)
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL)

    forecast_parser = sub.add_parser("forecast", help="Predict decision timing from minute-precision lodgement time")
    forecast_parser.add_argument("--lodged-at", required=True, help="ISO datetime, e.g. 2026-09-24T15:30+08:00")
    forecast_parser.add_argument("--as-of", help="ISO datetime; defaults to now")
    forecast_parser.add_argument("--timezone", default="Asia/Shanghai", help="Used only for naive datetimes")
    forecast_parser.add_argument("--cases", type=Path, default=DEFAULT_DATASET)
    forecast_parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    forecast_parser.add_argument("--official-flows", type=Path, default=DEFAULT_OFFICIAL_MONTHLY)
    forecast_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    forecast_parser.add_argument(
        "--engine",
        choices=["deployed", "anchor"],
        default="deployed",
        help="deployed uses calibrated v3; anchor keeps the conservative official P50/P90 method",
    )
    forecast_parser.add_argument("--education-level", default="PhD")
    forecast_parser.add_argument("--study-sector", default="Postgraduate Research")
    forecast_parser.add_argument("--submit-location", default="Outside Australia")
    forecast_parser.add_argument("--includes-partner", default="unknown", choices=["yes", "no", "unknown"])
    forecast_parser.add_argument("--is-diy", default="unknown", choices=["yes", "no", "unknown"])
    forecast_parser.add_argument("--provider-group", default="Unknown")
    forecast_parser.add_argument("--field-of-study-group", default="Unknown")

    visualize = sub.add_parser("visualize", help="Generate source-backed SVG evidence charts")
    visualize.add_argument("--cases", type=Path, default=DEFAULT_DATASET)
    visualize.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    visualize.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    visualize.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "collect":
        result = collect_visadashboard_to_csv(
            output=args.output,
            snapshot_date=args.snapshot_date,
            page_size=args.page_size,
            max_pages=args.max_pages,
            respect_robots=not args.ignore_robots,
        )
    elif args.command == "validate":
        result = validate_case_csv(args.input)
    elif args.command == "profile":
        result = profile_case_csv(args.input, args.output)
    elif args.command == "predict":
        result = predict_from_csv(args.input, args.education_level, args.submit_location)
    elif args.command == "social-collect":
        if args.source == "x":
            rows = collect_x_recent(args.query, max_pages=args.max_pages)
        else:
            urls = args.input.read_text(encoding="utf-8").splitlines()
            rows = collect_public_urls(urls, respect_robots=not args.ignore_robots)
        result = write_social_csv(args.output, rows)
    elif args.command == "official-refresh":
        if bool(args.lodged_xlsx) != bool(args.granted_xlsx):
            raise ValueError("Provide both --lodged-xlsx and --granted-xlsx, or neither")
        resources = None
        if args.lodged_xlsx and args.granted_xlsx:
            resources = {
                "lodged": ResourceInfo(
                    resource_id="ef31b2b4-a894-484b-99bc-e35d62ace777",
                    name="Student visas lodged pivot table at 2026-08-31",
                    url=f"{PACKAGE_URL}/resource/ef31b2b4-a894-484b-99bc-e35d62ace777",
                    last_modified="2026-09-23",
                ),
                "granted": ResourceInfo(
                    resource_id="dfc7a893-0523-4b8e-bc5a-829e35bec90f",
                    name="Student visas granted pivot table at 2026-08-31",
                    url=f"{PACKAGE_URL}/resource/dfc7a893-0523-4b8e-bc5a-829e35bec90f",
                    last_modified="2026-09-23",
                ),
            }
        result = refresh_official_monthly_flows(
            output=args.output,
            snapshots_dir=args.snapshots_dir,
            revisions_path=args.revisions,
            lodged_workbook=args.lodged_xlsx,
            granted_workbook=args.granted_xlsx,
            resources=resources,
        )
    elif args.command == "audit":
        result = write_audit_report(args.input, args.json_output, args.markdown_output)
    elif args.command == "train":
        artifact = train_survival_baselines(
            cases_path=args.cases,
            official_flows_path=args.official_flows,
            output=args.output,
        )
        result = {
            "output": str(args.output),
            "model_version": artifact["model_version"],
            "selected_baseline": artifact["selected_baseline"],
            "deployment_engine": artifact["deployment"]["engine"],
            "deployment_confidence": artifact["deployment_confidence"],
            "development_window": artifact["time_split"]["selection_and_calibration_window"],
            "final_holdout_window": artifact["time_split"]["final_holdout_window"],
            "final_holdout_evaluation": artifact["deployment_holdout_evaluation"],
            "calibration_candidate_decision": artifact["deployment"]["calibration_decision"],
        }
    elif args.command == "forecast":
        common = {
            "lodged_at": args.lodged_at,
            "as_of": args.as_of,
            "timezone_name": args.timezone,
            "education_level": args.education_level,
            "study_sector": args.study_sector,
            "submit_location": args.submit_location,
            "includes_partner": args.includes_partner,
            "is_diy": args.is_diy,
            "provider_group": args.provider_group,
            "field_of_study_group": args.field_of_study_group,
        }
        if args.engine == "anchor":
            result = forecast(
                cases_path=args.cases,
                official_path=args.official,
                model_artifact=args.model,
                **common,
            )
        else:
            result = forecast_survival_model(
                model_artifact=args.model,
                official_flows_path=args.official_flows,
                **common,
            )
    else:
        result = render_all(args.cases, args.official, args.model, args.output_dir)
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
