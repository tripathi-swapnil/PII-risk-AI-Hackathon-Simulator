from typing import List, Dict, Tuple
from dataclasses import dataclass

# -----------------------------
# Data Structures
# -----------------------------

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

# -----------------------------
# Risk Engine
# -----------------------------

SEVERITY_MAP = {
    "NAME": 1,
    "EMAIL": 1,
    "PHONE": 2,
    "ADDRESS": 2,
    "DOB": 3,
    "AADHAAR": 4,
    "SSN": 4,
    "BANK_ACCOUNT": 4
}

CRITICAL_LABELS = {"AADHAAR", "SSN", "BANK_ACCOUNT"}


def compute_document_risk(entities: List[Entity]) -> str:
    if any(e.label in CRITICAL_LABELS for e in entities):
        return "high"

    max_severity = max([SEVERITY_MAP[e.label] for e in entities], default=0)

    if max_severity <= 1:
        return "low"
    elif max_severity == 2:
        return "medium"
    else:
        return "high"


def compute_risk_score(entities: List[Entity], missed_critical: int) -> float:
    severity_sum = sum(SEVERITY_MAP[e.label] for e in entities)
    return severity_sum + (5 * missed_critical)


# -----------------------------
# Matching Utility
# -----------------------------

def match_entities(pred: List[Entity], gold: List[Entity]) -> Tuple[int, int, int]:
    """
    Returns (TP, partial, FP)
    """
    matched = set()
    tp, partial = 0, 0

    for p in pred:
        found = False
        for i, g in enumerate(gold):
            if i in matched:
                continue
            if p.start == g.start and p.end == g.end:
                matched.add(i)
                found = True
                if p.label == g.label:
                    tp += 1
                else:
                    partial += 1
                break
        if not found:
            pass

    fp = len(pred) - (tp + partial)
    return tp, partial, fp


# -----------------------------
# Detection Grader
# -----------------------------

def detection_score(pred: List[Entity], gold: List[Entity]) -> float:
    tp, partial, fp = match_entities(pred, gold)
    gt = len(gold)

    score = (tp + 0.5 * partial - 0.5 * fp) / max(1, gt)
    return max(0.0, min(1.0, score))


# -----------------------------
# Classification Grader
# -----------------------------

def classification_score(pred: List[Entity], gold: List[Entity], pred_risk: str, gold_risk: str) -> float:
    det_score = detection_score(pred, gold)

    risk_score = 1.0 if pred_risk == gold_risk else 0.0

    gold_critical = [g for g in gold if g.label in CRITICAL_LABELS]
    pred_critical_spans = {(p.start, p.end) for p in pred if p.label in CRITICAL_LABELS}

    critical_detected = all((g.start, g.end) in pred_critical_spans for g in gold_critical)
    critical_score = 1.0 if critical_detected else 0.0

    final = 0.4 * det_score + 0.3 * risk_score + 0.3 * critical_score
    return max(0.0, min(1.0, final))


# -----------------------------
# Redaction Utilities
# -----------------------------

def check_leakage(redacted_text: str, gold: List[Entity]) -> bool:
    for g in gold:
        if g.text in redacted_text:
            return True
    return False


def compute_coverage(redacted_text: str, gold: List[Entity]) -> float:
    covered = 0
    for g in gold:
        placeholder = f"[REDACTED_{g.label}]"
        if placeholder in redacted_text:
            covered += 1
    return covered / max(1, len(gold))


def format_preservation(original: str, redacted: str, gold: List[Entity]) -> float:
    # simple heuristic: length similarity
    return 1 - abs(len(original) - len(redacted)) / max(1, len(original))


# -----------------------------
# Redaction Grader
# -----------------------------

def redaction_score(original_text: str, redacted_text: str, gold: List[Entity]) -> float:
    coverage = compute_coverage(redacted_text, gold)
    leak_free = 0.0 if check_leakage(redacted_text, gold) else 1.0
    format_score = format_preservation(original_text, redacted_text, gold)

    score = 0.5 * coverage + 0.3 * leak_free + 0.2 * format_score
    return max(0.0, min(1.0, score))


# -----------------------------
# Hard Task Grader (Full)
# -----------------------------

def hard_task_score(pred: List[Entity], gold: List[Entity], pred_risk: str, gold_risk: str,
                    original_text: str, redacted_text: str, steps_used: int) -> float:

    det = detection_score(pred, gold)
    cls = 1.0 if pred_risk == gold_risk else 0.0
    red = redaction_score(original_text, redacted_text, gold)

    efficiency = max(0.0, 1 - (steps_used / 15))

    score = 0.3 * det + 0.2 * cls + 0.4 * red + 0.1 * efficiency

    # CRITICAL CAP
    gold_critical = [g for g in gold if g.label in CRITICAL_LABELS]
    pred_spans = {(p.start, p.end) for p in pred}

    missed_critical = any((g.start, g.end) not in pred_spans for g in gold_critical)

    if missed_critical:
        score = min(score, 0.4)

    return max(0.0, min(1.0, score))

