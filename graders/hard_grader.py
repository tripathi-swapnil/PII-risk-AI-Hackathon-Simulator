from env.state import Entity, RiskLevel
from graders.logic_grader import hard_task_score


def grade_hard(
    pred_entities: list[Entity],
    gold_entities: list[Entity],
    pred_risk: RiskLevel | str,
    gold_risk: RiskLevel | str,
    original_text: str,
    redacted_text: str,
    steps_used: int,
) -> float:
    return hard_task_score(
        pred_entities,
        gold_entities,
        pred_risk,
        gold_risk,
        original_text,
        redacted_text,
        steps_used,
    )
