from env.action import ActionType
from env.risk_engine import compute_risk_score
from env.state import Entity


def compute_reward(
    action_type: ActionType,
    base_reward: float,
    gold_entities: list[Entity],
    missed_critical: int,
) -> float:
    # Risk-calibrated shaping from the design doc:
    # reward = base_reward - (0.3 * risk_score)
    risk_score = compute_risk_score(gold_entities, missed_critical)
    reward = base_reward
    if action_type in {ActionType.DETECT, ActionType.CLASSIFY, ActionType.REDACT}:
        reward = base_reward - (0.3 * risk_score / max(1, len(gold_entities)))
    return round(reward, 4)
