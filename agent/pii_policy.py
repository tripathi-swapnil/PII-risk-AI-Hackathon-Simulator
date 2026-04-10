import re
from typing import Any

from env.risk_engine import compute_document_risk
from env.state import Entity, TaskType


_MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_AADHAAR_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}\b")
_DOB_RE = re.compile(rf"\b\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}\b")
_PHONE_RE = re.compile(r"\b(?:\+\d{1,3}[\s-]?)?\d(?:[\d\s-]{8,}\d)\b")
_BANK_RE = re.compile(r"\b(?:[A-Z]{2,8}\d{8,18}|\d{12,18})\b")
_ADDRESS_RE = re.compile(
    r"\b\d{1,4}\s+[A-Z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*){0,5}\s(?:Street|St|Road|Rd|Ave|Avenue|Lane|Drive|Colony|Terrace|Apartments)\b(?:,\s*[A-Za-z]+)?"
)
_NAME_CUE_RE = re.compile(
    r"\b(?:Name|Employee|Patient|owner|borrower|consultant|client|holder|contact(?:\s+is)?|signed\s+by)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"
)

_LABEL_PRIORITY = {
    "SSN": 0,
    "AADHAAR": 1,
    "BANK_ACCOUNT": 2,
    "EMAIL": 3,
    "PHONE": 4,
    "DOB": 5,
    "ADDRESS": 6,
    "NAME": 7,
}


def _add_matches(candidates: list[dict[str, Any]], text: str, regex: re.Pattern[str], label: str, confidence: float) -> None:
    for match in regex.finditer(text):
        start, end = match.start(), match.end()
        if label == "NAME" and match.lastindex:
            start, end = match.start(1), match.end(1)
        if start >= end:
            continue
        candidates.append(
            {
                "text": text[start:end],
                "label": label,
                "start": start,
                "end": end,
                "confidence": confidence,
            }
        )


def _has_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (a["end"] <= b["start"] or a["start"] >= b["end"])


def _is_probable_phone(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 10 <= len(digits) <= 15


def extract_entities(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    _add_matches(candidates, text, _SSN_RE, "SSN", 0.99)
    _add_matches(candidates, text, _AADHAAR_RE, "AADHAAR", 0.99)
    _add_matches(candidates, text, _EMAIL_RE, "EMAIL", 0.98)
    _add_matches(candidates, text, _DOB_RE, "DOB", 0.95)
    _add_matches(candidates, text, _ADDRESS_RE, "ADDRESS", 0.9)
    _add_matches(candidates, text, _NAME_CUE_RE, "NAME", 0.88)

    for match in _PHONE_RE.finditer(text):
        value = match.group(0)
        if _is_probable_phone(value):
            candidates.append(
                {
                    "text": value,
                    "label": "PHONE",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9,
                }
            )

    for match in _BANK_RE.finditer(text):
        value = match.group(0)
        if _SSN_RE.fullmatch(value) or _AADHAAR_RE.fullmatch(value):
            continue
        if _is_probable_phone(value):
            continue
        candidates.append(
            {
                "text": value,
                "label": "BANK_ACCOUNT",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.93,
            }
        )

    ordered = sorted(
        candidates,
        key=lambda item: (item["start"], _LABEL_PRIORITY.get(item["label"], 99), -(item["end"] - item["start"])),
    )

    selected: list[dict[str, Any]] = []
    for candidate in ordered:
        if any(_has_overlap(candidate, existing) for existing in selected):
            continue
        selected.append(candidate)

    return selected


def _risk_from_entities(entities: list[dict[str, Any]]) -> str:
    typed_entities = [
        Entity(
            text=ent["text"],
            label=ent["label"],
            start=ent["start"],
            end=ent["end"],
            confidence=float(ent.get("confidence", 0.7)),
        )
        for ent in entities
    ]
    return compute_document_risk(typed_entities).value


def redact_text(document_text: str, entities: list[dict[str, Any]]) -> str:
    redacted = document_text
    for ent in sorted(entities, key=lambda item: item["start"], reverse=True):
        redaction_token = f"[REDACTED_{ent['label']}]"
        redacted = redacted[: ent["start"]] + redaction_token + redacted[ent["end"] :]
    return redacted


def build_rule_based_action(
    task_type: TaskType | str,
    step_idx: int,
    document_text: str,
    detected_entities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    task = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    entities = detected_entities or extract_entities(document_text)

    if step_idx == 0:
        return {
            "action_type": "detect",
            "spans": [{"start": ent["start"], "end": ent["end"], "label": ent["label"]} for ent in entities],
            "confidence": 0.85,
            "reasoning": "rule_based_detection",
        }

    if task in {"medium", "hard"} and step_idx == 1:
        return {
            "action_type": "classify",
            "classification": _risk_from_entities(entities),
            "confidence": 0.82,
            "reasoning": "rule_based_risk_classification",
        }

    if task == "hard" and step_idx == 2:
        return {
            "action_type": "redact",
            "redacted_text": redact_text(document_text, entities),
            "confidence": 0.8,
            "reasoning": "rule_based_redaction",
        }

    return {"action_type": "finalize", "confidence": 0.95, "reasoning": "rule_based_finalize"}