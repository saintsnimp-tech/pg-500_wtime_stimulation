from __future__ import annotations

import urllib.parse
from datetime import date
from typing import Iterator

from .http import PoliteJsonClient
from .schema import NormalizedCase, normalize_visadashboard_record


VISADASHBOARD_PAGE = "https://www.visadashboard.top/dashboard/table"
VISADASHBOARD_API = "https://www.visadashboard.top/api/visaTable"


def collect_visadashboard(
    client: PoliteJsonClient,
    snapshot_date: date,
    page_size: int = 100,
    max_pages: int | None = None,
) -> Iterator[NormalizedCase]:
    """Collect public table rows without retaining IDs, officer codes or free text."""
    page = 1
    total_pages = 1
    while page <= total_pages and (max_pages is None or page <= max_pages):
        query = urllib.parse.urlencode(
            {
                "pagination[current]": page,
                "pagination[pageSize]": page_size,
                "sortField": "getVisaTime",
                "sortOrder": "descend",
            }
        )
        payload = client.get_json(f"{VISADASHBOARD_API}?{query}")
        records = payload.get("data")
        if not isinstance(records, list):
            raise ValueError("VisaDashboard response does not contain a data list")
        for raw in records:
            if isinstance(raw, dict):
                yield normalize_visadashboard_record(raw, snapshot_date, VISADASHBOARD_PAGE)
        total_pages = int(payload.get("totalPages") or 1)
        page += 1

