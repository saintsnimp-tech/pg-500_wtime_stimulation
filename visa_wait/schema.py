from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


SCHEMA_FIELDS = [
    "record_id",
    "duplicate_key",
    "source_id",
    "source_quality_tier",
    "snapshot_date",
    "visa_subclass",
    "application_status",
    "event_observed",
    "lodgement_date",
    "decision_date",
    "as_of_date",
    "waiting_days",
    "education_level",
    "study_sector",
    "field_of_study_group",
    "submit_location",
    "includes_partner",
    "is_diy",
    "provider_group",
    "record_quality_score",
    "quality_flags",
    "provenance_url",
]


@dataclass(frozen=True)
class NormalizedCase:
    record_id: str
    duplicate_key: str
    source_id: str
    source_quality_tier: str
    snapshot_date: str
    visa_subclass: str
    application_status: str
    event_observed: int
    lodgement_date: str
    decision_date: str
    as_of_date: str
    waiting_days: int | None
    education_level: str
    study_sector: str
    field_of_study_group: str
    submit_location: str
    includes_partner: str
    is_diy: str
    provider_group: str
    record_quality_score: int
    quality_flags: str
    provenance_url: str

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _iso_date(value: Any) -> date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _boolish(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "是", "有", "diy"}:
        return "yes"
    if text in {"false", "0", "no", "否", "无"}:
        return "no"
    return "unknown"


def _education(value: Any) -> tuple[str, str]:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("博士", "phd", "doctoral", "doctorate")):
        return "PhD", "Postgraduate Research"
    if any(token in text for token in ("硕士", "master", "研究生")):
        return "Master", "Higher Education"
    if any(token in text for token in ("本科", "bachelor", "undergraduate")):
        return "Bachelor", "Higher Education"
    if any(token in text for token in ("中学", "school")):
        return "School", "Schools"
    if any(token in text for token in ("语言", "elicos", "english")):
        return "ELICOS", "Independent ELICOS"
    if any(token in text for token in ("职业", "vet", "tafe")):
        return "VET", "Vocational Education and Training"
    return "Unknown", "Unknown"


def _field_group(value: Any) -> str:
    text = str(value or "").strip().lower()
    groups = [
        ("Health and Medicine", ("医", "health", "medicine", "medical", "nursing", "mdhs", "pharma")),
        ("Engineering", ("工程", "engineering", "civil", "mechanical", "electrical", "材料")),
        ("Computing and Data", ("计算机", "computer", "software", "data", "统计", "ai", "人工智能")),
        ("Natural Sciences", ("biology", "microbiology", "chem", "physics", "science", "生物", "化学", "物理")),
        ("Business and Economics", ("商", "business", "finance", "account", "econom", "management")),
        ("Education", ("教育", "education", "teaching")),
        ("Humanities and Social Sciences", ("人文", "社科", "social", "humanit", "law", "法律", "arts")),
        ("Built Environment", ("建筑", "architecture", "construction", "urban", "planning")),
    ]
    if not text:
        return "Unknown"
    for name, tokens in groups:
        if any(token in text for token in tokens):
            return name
    return "Other"


def _location(value: Any) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("国内", "境外", "offshore", "outside")):
        return "Outside Australia"
    if any(token in text for token in ("澳洲", "境内", "onshore", "inside")):
        return "In Australia"
    return "Unknown"


def _partner(value: Any) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("携", "配偶", "couple", "partner", "副申")):
        return "yes"
    if any(token in text for token in ("单独", "single", "alone")):
        return "no"
    return _boolish(value)


def _provider_group(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not text:
        return "Unknown"
    go8_tokens = (
        "monash", "melbourne", "unsw", "sydney", "queensland", "uq", "anu",
        "australian national", "adelaide", "western australia", "uwa", "go8", "八大",
    )
    return "Group of Eight" if any(token in text for token in go8_tokens) else "Other/Unverified"


def normalize_visadashboard_record(
    raw: dict[str, Any], snapshot_date: date, provenance_url: str
) -> NormalizedCase:
    lodged = _iso_date(raw.get("submitTime"))
    decision = _iso_date(raw.get("getVisaTime"))
    claimed_granted = _boolish(raw.get("ifGetVisa")) == "yes"
    event_observed = int(bool(claimed_granted and decision))
    endpoint = decision if event_observed else snapshot_date
    flags: list[str] = []
    score = 100

    if lodged is None:
        flags.append("missing_lodgement_date")
        score -= 55
    if claimed_granted and decision is None:
        flags.append("granted_without_decision_date")
        score -= 30
    if decision and lodged and decision < lodged:
        flags.append("decision_before_lodgement")
        score -= 60
    if lodged and lodged > snapshot_date:
        flags.append("future_lodgement")
        score -= 60

    waiting_days = (endpoint - lodged).days if lodged else None
    if waiting_days is not None and (waiting_days < 0 or waiting_days > 3650):
        flags.append("implausible_waiting_days")
        score -= 50

    education_level, study_sector = _education(raw.get("educationLevel"))
    if education_level == "Unknown":
        flags.append("unknown_education_level")
        score -= 8

    submit_location = _location(raw.get("submitPlace"))
    if submit_location == "Unknown":
        flags.append("unknown_submit_location")
        score -= 5

    stable_source_id = str(raw.get("_id") or "")
    if not stable_source_id:
        stable_source_id = json.dumps(raw, sort_keys=True, ensure_ascii=False)
        flags.append("missing_source_record_id")
        score -= 5
    record_id = "VD-" + hashlib.sha256(stable_source_id.encode("utf-8")).hexdigest()[:20]

    education = education_level
    partner = _partner(raw.get("ifIncludedCouple"))
    diy = _boolish(raw.get("ifDIY"))
    field_group = _field_group(raw.get("major"))
    duplicate_material = "|".join(
        [str(lodged or ""), str(decision or ""), education, submit_location, partner, diy, field_group]
    )
    duplicate_key = hashlib.sha256(duplicate_material.encode("utf-8")).hexdigest()[:20]

    return NormalizedCase(
        record_id=record_id,
        duplicate_key=duplicate_key,
        source_id="PLAT-VISADASHBOARD",
        source_quality_tier="C",
        snapshot_date=snapshot_date.isoformat(),
        visa_subclass="500",
        application_status="granted" if event_observed else "pending",
        event_observed=event_observed,
        lodgement_date=lodged.isoformat() if lodged else "",
        decision_date=decision.isoformat() if event_observed and decision else "",
        as_of_date=(decision if event_observed and decision else snapshot_date).isoformat(),
        waiting_days=waiting_days,
        education_level=education,
        study_sector=study_sector,
        field_of_study_group=field_group,
        submit_location=submit_location,
        includes_partner=partner,
        is_diy=diy,
        provider_group=_provider_group(raw.get("schoolType")),
        record_quality_score=max(0, score),
        quality_flags="|".join(flags),
        provenance_url=provenance_url,
    )

