from env.state import Entity
from graders.logic_grader import detection_score


def grade_easy(pred_entities: list[Entity], gold_entities: list[Entity]) -> float:
    return detection_score(pred_entities, gold_entities)
