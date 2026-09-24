from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from .http import HttpPolicy, PoliteJsonClient


SOCIAL_FIELDS = [
    "evidence_id",
    "duplicate_key",
    "platform",
    "source_url",
    "published_at",
    "lodgement_at",
    "decision_at",
    "application_status",
    "event_observed",
    "waiting_minutes",
    "waiting_days",
    "education_level",
    "study_sector",
    "submit_location",
    "source_quality_tier",
    "record_quality_score",
    "parser_confidence",
    "requires_manual_review",
    "collected_at",
    "content_sha256",
]


@dataclass(frozen=True)
class SocialEvidence:
    evidence_id: str
    duplicate_key: str
    platform: str
    source_url: str
    published_at: str
    lodgement_at: str
    decision_at: str
    application_status: str
    event_observed: int
    waiting_minutes: int | None
    waiting_days: int | None
    education_level: str
    study_sector: str
    submit_location: str
    source_quality_tier: str
    record_quality_score: int
    parser_confidence: str
    requires_manual_review: str
    collected_at: str
    content_sha256: str

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self._json_ld = False
        self.json_ld_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta":
            key = values.get("property") or values.get("name")
            if key and values.get("content"):
                self.meta[key.lower()] = values["content"]
        if tag.lower() == "script" and values.get("type", "").lower() == "application/ld+json":
            self._json_ld = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script":
            self._json_ld = False

    def handle_data(self, data: str) -> None:
        if self._json_ld:
            self.json_ld_chunks.append(data)


DATE_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*(?:[-/.年])\s*(?P<month>\d{1,2})\s*(?:[-/.月])\s*(?P<day>\d{1,2})\s*日?"
)
LODGEMENT_LABELS = ("递签", "提交", "申请", "lodged", "lodgement", "submitted", "applied")
DECISION_LABELS = ("下签", "获签", "批准", "granted", "grant", "approved", "decision")


def _find_labelled_date(text: str, labels: tuple[str, ...]) -> date | None:
    lowered = text.lower()
    matches: list[tuple[int, date]] = []
    date_matches = list(DATE_PATTERN.finditer(lowered))
    label_matches = [
        (match.start(), match.end())
        for label in labels
        for match in re.finditer(re.escape(label), lowered)
    ]
    for match in date_matches:
        try:
            parsed = date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
        except ValueError:
            continue
        distance = min(
            (min(abs(start - match.end()), abs(match.start() - end)) for start, end in label_matches),
            default=10_000,
        )
        if distance <= 50:
            matches.append((distance, parsed))
    return min(matches, default=(0, None), key=lambda item: item[0])[1]


def extract_timeline(text: str) -> tuple[date | None, date | None, str]:
    cleaned = html.unescape(re.sub(r"\s+", " ", text or "")).strip()
    lodged = _find_labelled_date(cleaned, LODGEMENT_LABELS)
    decision = _find_labelled_date(cleaned, DECISION_LABELS)
    if lodged and decision and decision >= lodged:
        return lodged, decision, "high"
    if lodged or decision:
        return lodged, decision, "low"
    return None, None, "none"


def _broad_features(text: str) -> tuple[str, str, str]:
    lowered = text.lower()
    if any(token in lowered for token in ("博士", "phd", "doctoral")):
        education, sector = "PhD", "Postgraduate Research"
    elif any(token in lowered for token in ("硕士", "master")):
        education, sector = "Master", "Higher Education"
    elif any(token in lowered for token in ("本科", "bachelor")):
        education, sector = "Bachelor", "Higher Education"
    else:
        education, sector = "Unknown", "Unknown"
    if any(token in lowered for token in ("国内递交", "境外递交", "offshore", "outside australia")):
        location = "Outside Australia"
    elif any(token in lowered for token in ("澳洲境内", "境内递交", "onshore", "in australia")):
        location = "In Australia"
    else:
        location = "Unknown"
    return education, sector, location


def normalize_social_text(
    *, platform: str, source_url: str, text: str, published_at: str = "", collected_at: datetime | None = None
) -> SocialEvidence | None:
    lodged, decision, parser_confidence = extract_timeline(text)
    if not lodged:
        return None
    collected = collected_at or datetime.now(timezone.utc)
    event = int(decision is not None and decision >= lodged)
    end = decision or collected.date()
    waiting_days = (end - lodged).days
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    evidence_id = "SOC-" + hashlib.sha256(f"{platform}|{source_url}".encode("utf-8")).hexdigest()[:20]
    education, sector, location = _broad_features(text)
    duplicate_material = f"{lodged}|{decision or ''}|{education}|{location}"
    duplicate_key = hashlib.sha256(duplicate_material.encode("utf-8")).hexdigest()[:20]
    score = 75 if event else 60
    if parser_confidence != "high":
        score -= 25
    if education == "Unknown":
        score -= 5
    return SocialEvidence(
        evidence_id=evidence_id,
        duplicate_key=duplicate_key,
        platform=platform,
        source_url=source_url,
        published_at=published_at,
        lodgement_at=lodged.isoformat(),
        decision_at=decision.isoformat() if decision else "",
        application_status="granted" if event else "pending_or_unknown",
        event_observed=event,
        waiting_minutes=waiting_days * 1440,
        waiting_days=waiting_days,
        education_level=education,
        study_sector=sector,
        submit_location=location,
        source_quality_tier="D",
        record_quality_score=max(0, score),
        parser_confidence=parser_confidence,
        requires_manual_review="no" if parser_confidence == "high" else "yes",
        collected_at=collected.replace(microsecond=0).isoformat(),
        content_sha256=content_hash,
    )


def _page_text(source: str) -> tuple[str, str]:
    parser = _MetadataParser()
    parser.feed(source)
    chunks = [
        parser.meta.get("og:title", ""),
        parser.meta.get("og:description", ""),
        parser.meta.get("description", ""),
        *parser.json_ld_chunks,
    ]
    published = parser.meta.get("article:published_time", "")
    return " ".join(filter(None, chunks)), published


def platform_from_url(url: str) -> str:
    host = urllib.parse.urlsplit(url).netloc.lower()
    if "xiaohongshu" in host or "xhslink" in host:
        return "Xiaohongshu"
    if "weibo" in host:
        return "Weibo"
    if host.endswith("x.com") or "twitter.com" in host:
        return "X"
    if "reddit.com" in host:
        return "Reddit"
    return "PublicWeb"


def collect_public_urls(
    urls: Iterable[str], *, respect_robots: bool = True
) -> list[SocialEvidence]:
    client = PoliteJsonClient(HttpPolicy(respect_robots=respect_robots, min_delay_seconds=1.0))
    results: list[SocialEvidence] = []
    for url in urls:
        clean_url = url.strip()
        if not clean_url or clean_url.startswith("#"):
            continue
        source = client.get_text(clean_url, headers={"Accept": "text/html,application/xhtml+xml"})
        text, published = _page_text(source)
        record = normalize_social_text(
            platform=platform_from_url(clean_url), source_url=clean_url, text=text, published_at=published
        )
        if record:
            results.append(record)
    return results


def collect_x_recent(
    query: str, *, bearer_token: str | None = None, max_pages: int = 3
) -> list[SocialEvidence]:
    token = bearer_token or os.environ.get("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError("X_BEARER_TOKEN is required for the official X API adapter")
    client = PoliteJsonClient(HttpPolicy(respect_robots=False, min_delay_seconds=1.0))
    results: list[SocialEvidence] = []
    next_token = ""
    for _ in range(max_pages):
        params = {
            "query": query,
            "max_results": 100,
            "tweet.fields": "created_at,lang",
        }
        if next_token:
            params["next_token"] = next_token
        url = "https://api.x.com/2/tweets/search/recent?" + urllib.parse.urlencode(params)
        payload = client.get_json(url, headers={"Authorization": f"Bearer {token}"})
        for item in payload.get("data", []):
            if not isinstance(item, dict):
                continue
            tweet_id = str(item.get("id", ""))
            record = normalize_social_text(
                platform="X",
                source_url=f"https://x.com/i/web/status/{tweet_id}",
                text=str(item.get("text", "")),
                published_at=str(item.get("created_at", "")),
            )
            if record:
                results.append(record)
        next_token = str((payload.get("meta") or {}).get("next_token") or "")
        if not next_token:
            break
    return results


def write_social_csv(path: Path, rows: Iterable[SocialEvidence]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = {row["evidence_id"]: row for row in csv.DictReader(handle) if row.get("evidence_id")}
    for row in rows:
        existing[row.evidence_id] = row.as_row()
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOCIAL_FIELDS)
        writer.writeheader()
        writer.writerows(existing.values())
    return {"output": str(path), "rows": len(existing)}
