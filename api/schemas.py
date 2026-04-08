from typing import Any

from pydantic import BaseModel, Field

from env.action import Action, ActionType
from env import Reward
from env.state import Entity, Observation, TaskType


class ResetRequest(BaseModel):
    task_type: TaskType = TaskType.EASY


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


class TaskInfo(BaseModel):
    name: TaskType
    allowed_actions: list[ActionType]
    max_steps: int


class GraderRequest(BaseModel):
    task_type: TaskType
    pred_entities: list[Entity] = Field(default_factory=list)
    pred_risk: str | None = None
    redacted_text: str | None = None
    steps_used: int = 0


class GraderResponse(BaseModel):
    score: float


class BaselineResponse(BaseModel):
    scores: dict[str, float]


class BaselineRequest(BaseModel):
    use_ai: bool = True
