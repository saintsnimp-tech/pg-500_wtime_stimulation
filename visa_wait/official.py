from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import xml.parsers.expat as expat
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PACKAGE_ID = "324aa4f7-46bb-4d56-bc2d-772333a2317e"
PACKAGE_API = f"https://data.gov.au/data/api/3/action/package_show?id={PACKAGE_ID}"
PACKAGE_URL = f"https://data.gov.au/data/dataset/{PACKAGE_ID}"
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

APPLICANT_TYPE = "Primary"
CLIENT_LOCATION = "Outside Australia"
CITIZENSHIP_COUNTRY = "China, Peoples Republic of (excl SARs)"
SECTOR = "Postgraduate Research Sector"

FLOW_FIELDS = [
    "source_snapshot_date",
    "activity_month",
    "activity_type",
    "count",
    "applicant_type",
    "client_location",
    "citizenship_country",
    "sector",
    "source_resource_id",
    "source_last_modified",
    "source_file_sha256",
    "source_url",
    "extraction_method",
]

REVISION_FIELDS = [
    "detected_at",
    "prior_snapshot_date",
    "new_snapshot_date",
    "activity_month",
    "activity_type",
    "prior_count",
    "new_count",
    "difference",
    "prior_source_sha256",
    "new_source_sha256",
]


@dataclass(frozen=True)
class ResourceInfo:
    resource_id: str
    name: str
    url: str
    last_modified: str
    format: str = "XLSX"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_package_resources() -> dict[str, ResourceInfo]:
    request = urllib.request.Request(
        PACKAGE_API,
        headers={"User-Agent": "pg-500-wtime-stimulation/0.4 (+public statistical research)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not payload.get("success"):
        raise ValueError("data.gov.au package metadata request was not successful")
    selected: dict[str, ResourceInfo] = {}
    for item in payload["result"].get("resources", []):
        name = str(item.get("name") or "")
        lowered = name.lower()
        activity_type = "lodged" if "student visas lodged report" in lowered else "granted" if "student visas granted report" in lowered else ""
        if not activity_type or str(item.get("format") or "").upper() != "XLSX":
            continue
        selected[activity_type] = ResourceInfo(
            resource_id=str(item.get("id") or ""),
            name=name,
            url=str(item.get("url") or ""),
            last_modified=str(item.get("last_modified") or ""),
            format="XLSX",
        )
    if set(selected) != {"lodged", "granted"}:
        raise ValueError("Could not locate both lodged and granted BP0015 XLSX resources")
    return selected


def download_resource(resource: ResourceInfo, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        resource.url,
        headers={"User-Agent": "pg-500-wtime-stimulation/0.4 (+public statistical research)"},
    )
    with urllib.request.urlopen(request, timeout=180) as response, target.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    return target


def _shared_values(field: ET.Element) -> list[str | None]:
    node = field.find(XML_NS + "sharedItems")
    if node is None:
        return []
    return [child.attrib.get("v") for child in node]


def _snapshot_date(resource: ResourceInfo, financial_year_values: list[str | None]) -> str:
    embedded_dates = []
    for value in financial_year_values:
        match = re.search(r"to\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", str(value or ""))
        if match:
            embedded_dates.append(datetime.strptime(match.group(1), "%d %B %Y").date())
    if embedded_dates:
        return max(embedded_dates).isoformat()
    match = re.search(r"at\s+(\d{4}-\d{2}-\d{2})", resource.name)
    if match:
        return match.group(1)
    raise ValueError(f"Could not determine BP0015 snapshot date from {resource.name!r}")


def _calendar_month(financial_year: str, month_value: str) -> str:
    start_year = int(financial_year[:4])
    month_index = int(month_value[1:3])
    calendar_month = month_index + 6 if month_index <= 6 else month_index - 6
    calendar_year = start_year if month_index <= 6 else start_year + 1
    return f"{calendar_year:04d}-{calendar_month:02d}"


def extract_bp0015(
    workbook_path: Path,
    activity_type: str,
    resource: ResourceInfo,
    start_month: str = "2015-01",
) -> list[dict[str, object]]:
    if activity_type not in {"lodged", "granted"}:
        raise ValueError("activity_type must be lodged or granted")
    totals: defaultdict[str, int] = defaultdict(int)
    with zipfile.ZipFile(workbook_path) as archive:
        definition = ET.fromstring(archive.read("xl/pivotCache/pivotCacheDefinition1.xml"))
        fields_node = definition.find(XML_NS + "cacheFields")
        if fields_node is None:
            raise ValueError("BP0015 workbook does not contain pivot cache fields")
        fields = list(fields_node)
        names = [field.attrib.get("name", "") for field in fields]
        shared = [_shared_values(field) for field in fields]

        financial_year_index = next(
            index for index, name in enumerate(names) if name.startswith("Financial Year of Visa")
        )
        month_index = names.index("Month")
        sector_index = names.index("Sector")
        applicant_index = names.index("Applicant Type")
        country_index = names.index("Citizenship Country")
        location_index = names.index("Client Location")
        total_index = names.index("Total")

        expected_indices = {
            sector_index: str(shared[sector_index].index(SECTOR)),
            applicant_index: str(shared[applicant_index].index(APPLICANT_TYPE)),
            country_index: str(shared[country_index].index(CITIZENSHIP_COUNTRY)),
            location_index: str(shared[location_index].index(CLIENT_LOCATION)),
        }
        state: dict[str, list[str | None] | None] = {"record": None}

        def start_element(name: str, attributes: dict[str, str]) -> None:
            if name == "r":
                state["record"] = []
            elif state["record"] is not None and name in {"x", "n", "m", "s", "d", "b", "e"}:
                state["record"].append(attributes.get("v"))

        def end_element(name: str) -> None:
            if name != "r":
                return
            values = state["record"]
            state["record"] = None
            if values is None or any(values[index] != expected for index, expected in expected_indices.items()):
                return
            financial_year = str(shared[financial_year_index][int(str(values[financial_year_index]))])
            month_value = str(shared[month_index][int(str(values[month_index]))])
            activity_month = _calendar_month(financial_year, month_value)
            if activity_month < start_month:
                return
            totals[activity_month] += int(float(values[total_index] or 0))

        parser = expat.ParserCreate()
        parser.StartElementHandler = start_element
        parser.EndElementHandler = end_element
        with archive.open("xl/pivotCache/pivotCacheRecords1.xml") as records:
            while True:
                chunk = records.read(1024 * 1024)
                if not chunk:
                    break
                parser.Parse(chunk, False)
            parser.Parse(b"", True)

        snapshot_date = _snapshot_date(resource, shared[financial_year_index])
    source_hash = file_sha256(workbook_path)
    return [
        {
            "source_snapshot_date": snapshot_date,
            "activity_month": month,
            "activity_type": activity_type,
            "count": totals[month],
            "applicant_type": APPLICANT_TYPE,
            "client_location": CLIENT_LOCATION,
            "citizenship_country": CITIZENSHIP_COUNTRY,
            "sector": SECTOR,
            "source_resource_id": resource.resource_id,
            "source_last_modified": resource.last_modified,
            "source_file_sha256": source_hash,
            "source_url": resource.url,
            "extraction_method": "BP0015 pivot-cache aggregation",
        }
        for month in sorted(totals)
        if month <= snapshot_date[:7]
    ]


def _atomic_write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)
        temp_name = handle.name
    os.replace(temp_name, path)
    return len(rows_list)


def validate_monthly_flows(rows: list[dict[str, object]]) -> dict[str, object]:
    issues: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    by_activity: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["source_snapshot_date"]),
            str(row["activity_month"]),
            str(row["activity_type"]),
        )
        if key in seen:
            issues.append("duplicate_grain:" + "|".join(key))
        seen.add(key)
        if int(row["count"]) < 0:
            issues.append("negative_count:" + "|".join(key))
        by_activity[str(row["activity_type"])].append(str(row["activity_month"]))
    if set(by_activity) != {"lodged", "granted"}:
        issues.append("missing_activity_type")
    for activity_type, months in by_activity.items():
        if len(months) != len(set(months)):
            issues.append(f"duplicate_month:{activity_type}")
        if months and (min(months) != "2015-01" or max(months) < "2026-08"):
            issues.append(f"unexpected_coverage:{activity_type}:{min(months)}:{max(months)}")
    return {
        "rows": len(rows),
        "unique_keys": len(seen),
        "activity_types": sorted(by_activity),
        "month_min": min((str(row["activity_month"]) for row in rows), default=""),
        "month_max": max((str(row["activity_month"]) for row in rows), default=""),
        "issues": issues,
    }


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _revision_rows(
    prior: list[dict[str, str]], current: list[dict[str, object]], detected_at: str
) -> list[dict[str, object]]:
    prior_map = {
        (row.get("activity_month", ""), row.get("activity_type", "")): row for row in prior
    }
    changes: list[dict[str, object]] = []
    for row in current:
        key = (str(row["activity_month"]), str(row["activity_type"]))
        old = prior_map.get(key)
        if old is None or int(old.get("count") or 0) == int(row["count"]):
            continue
        changes.append(
            {
                "detected_at": detected_at,
                "prior_snapshot_date": old.get("source_snapshot_date", ""),
                "new_snapshot_date": row["source_snapshot_date"],
                "activity_month": row["activity_month"],
                "activity_type": row["activity_type"],
                "prior_count": old.get("count", ""),
                "new_count": row["count"],
                "difference": int(row["count"]) - int(old.get("count") or 0),
                "prior_source_sha256": old.get("source_file_sha256", ""),
                "new_source_sha256": row["source_file_sha256"],
            }
        )
    return changes


def refresh_official_monthly_flows(
    *,
    output: Path,
    snapshots_dir: Path,
    revisions_path: Path,
    lodged_workbook: Path | None = None,
    granted_workbook: Path | None = None,
    resources: dict[str, ResourceInfo] | None = None,
) -> dict[str, object]:
    resource_map = resources or fetch_package_resources()
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    try:
        if lodged_workbook is None or granted_workbook is None:
            temporary_directory = tempfile.TemporaryDirectory(prefix="bp0015-")
            temp_root = Path(temporary_directory.name)
            lodged_workbook = download_resource(resource_map["lodged"], temp_root / "lodged.xlsx")
            granted_workbook = download_resource(resource_map["granted"], temp_root / "granted.xlsx")
        rows = extract_bp0015(lodged_workbook, "lodged", resource_map["lodged"])
        rows.extend(extract_bp0015(granted_workbook, "granted", resource_map["granted"]))
        rows.sort(key=lambda row: (str(row["activity_month"]), str(row["activity_type"])))
        validation = validate_monthly_flows(rows)
        if validation["issues"]:
            raise ValueError("Official monthly flow validation failed: " + ", ".join(validation["issues"]))

        prior_rows = _load_csv(output)
        detected_at = datetime.now().astimezone().replace(microsecond=0).isoformat()
        new_revisions = _revision_rows(prior_rows, rows, detected_at)
        existing_revisions = _load_csv(revisions_path)
        _atomic_write_csv(revisions_path, [*existing_revisions, *new_revisions], REVISION_FIELDS)
        _atomic_write_csv(output, rows, FLOW_FIELDS)
        snapshot_date = str(rows[0]["source_snapshot_date"])
        snapshot_path = snapshots_dir / f"official_monthly_flows_{snapshot_date}.csv"
        _atomic_write_csv(snapshot_path, rows, FLOW_FIELDS)
        return {
            "output": str(output),
            "snapshot_output": str(snapshot_path),
            "revisions_output": str(revisions_path),
            "new_revisions": len(new_revisions),
            "validation": validation,
            "resources": {name: asdict(resource) for name, resource in resource_map.items()},
        }
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
