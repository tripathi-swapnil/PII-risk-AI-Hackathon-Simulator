from typing import Iterable

from env.state import Entity, RiskLevel


SEVERITY_MAP = {
    "NAME": 1,
    "EMAIL": 1,
    "PHONE": 2,
    "ADDRESS": 2,
    "DOB": 3,
    "AADHAAR": 4,
    "SSN": 4,
    "BANK_ACCOUNT": 4,
}

CRITICAL_LABELS = {"AADHAAR", "SSN", "BANK_ACCOUNT"}


def normalize_label(label: str) -> str:
    return label.strip().upper()


def severity_for(label: str) -> int:
    return SEVERITY_MAP.get(normalize_label(label), 1)


def is_critical(label: str) -> bool:
    return normalize_label(label) in CRITICAL_LABELS


def compute_document_risk(entities: Iterable[Entity]) -> RiskLevel:
    entity_list = list(entities)
    if any(is_critical(e.label) for e in entity_list):
        return RiskLevel.HIGH

    max_severity = max((severity_for(e.label) for e in entity_list), default=0)
    if max_severity <= 1:
        return RiskLevel.LOW
    if max_severity == 2:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def count_missed_critical(pred: Iterable[Entity], gold: Iterable[Entity]) -> int:
    pred_spans = {(e.start, e.end) for e in pred}
    return sum(
        1
        for g in gold
        if is_critical(g.label) and (g.start, g.end) not in pred_spans
    )


def compute_risk_score(gold: Iterable[Entity], missed_critical: int) -> float:
    severity_sum = sum(severity_for(e.label) for e in gold)
    return float(severity_sum + (5 * missed_critical))


def constraint_violated(redacted_text: str | None, gold: Iterable[Entity]) -> bool:
    if redacted_text is None:
        return False
    for g in gold:
        if is_critical(g.label) and g.text in redacted_text:
            return True
    return False
