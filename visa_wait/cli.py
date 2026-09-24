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


DEFAULT_DATASET = Path("data/11_visadashboard_cases.csv")
DEFAULT_PROFILE = Path("data/12_visadashboard_profile_2026-09.csv")


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
    else:
        result = predict_from_csv(args.input, args.education_level, args.submit_location)
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
