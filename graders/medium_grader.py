from env.state import Entity, RiskLevel
from graders.logic_grader import classification_score


def grade_medium(
    pred_entities: list[Entity],
    gold_entities: list[Entity],
    pred_risk: RiskLevel | str,
    gold_risk: RiskLevel | str,
) -> float:
    return classification_score(pred_entities, gold_entities, pred_risk, gold_risk)
