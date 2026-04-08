from typing import Union

from env.risk_engine import is_critical
from env.state import Entity, RiskLevel


def normalize_risk(risk: Union[RiskLevel, str]) -> str:
    if isinstance(risk, RiskLevel):
        return risk.value
    if isinstance(risk, str):
        return risk.strip().lower()
    raise ValueError("Risk must be RiskLevel or string.")


def match_entities(pred: list[Entity], gold: list[Entity]) -> tuple[int, int, int]:
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
            continue
    fp = len(pred) - (tp + partial)
    return tp, partial, fp


def detection_score(pred: list[Entity], gold: list[Entity]) -> float:
    tp, partial, fp = match_entities(pred, gold)
    gt = len(gold)
    score = (tp + 0.5 * partial - 0.5 * fp) / max(1, gt)
    return max(0.0, min(1.0, score))


def classification_score(
    pred: list[Entity],
    gold: list[Entity],
    pred_risk: Union[RiskLevel, str],
    gold_risk: Union[RiskLevel, str],
) -> float:
    det_score = detection_score(pred, gold)
    risk_score = 1.0 if normalize_risk(pred_risk) == normalize_risk(gold_risk) else 0.0
    gold_critical = [g for g in gold if is_critical(g.label)]
    pred_critical_spans = {(p.start, p.end) for p in pred if is_critical(p.label)}
    critical_detected = all((g.start, g.end) in pred_critical_spans for g in gold_critical)
    critical_score = 1.0 if critical_detected else 0.0
    final = 0.4 * det_score + 0.3 * risk_score + 0.3 * critical_score
    return max(0.0, min(1.0, final))


def redaction_score(original_text: str, redacted_text: str, gold: list[Entity]) -> float:
    coverage = 0
    for g in gold:
        if f"[REDACTED_{g.label}]" in redacted_text or f"[REDACTED_{g.label.upper()}]" in redacted_text:
            coverage += 1
    coverage_score = coverage / max(1, len(gold))
    leak_free = 0.0 if any(g.text in redacted_text for g in gold) else 1.0
    format_score = 1 - abs(len(original_text) - len(redacted_text)) / max(1, len(original_text))
    score = 0.5 * coverage_score + 0.3 * leak_free + 0.2 * format_score
    return max(0.0, min(1.0, score))


def hard_task_score(
    pred: list[Entity],
    gold: list[Entity],
    pred_risk: Union[RiskLevel, str],
    gold_risk: Union[RiskLevel, str],
    original_text: str,
    redacted_text: str,
    steps_used: int,
) -> float:
    det = detection_score(pred, gold)
    cls = 1.0 if normalize_risk(pred_risk) == normalize_risk(gold_risk) else 0.0
    red = redaction_score(original_text, redacted_text, gold)
    efficiency = max(0.0, 1 - (steps_used / 15))
    score = 0.3 * det + 0.2 * cls + 0.4 * red + 0.1 * efficiency

    gold_critical = [g for g in gold if is_critical(g.label)]
    pred_spans = {(p.start, p.end) for p in pred}
    missed_critical = any((g.start, g.end) not in pred_spans for g in gold_critical)
    if missed_critical:
        score = min(score, 0.4)
    return max(0.0, min(1.0, score))
