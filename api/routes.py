from fastapi import APIRouter

from api.schemas import (
    BaselineRequest,
    BaselineResponse,
    GraderRequest,
    GraderResponse,
    ResetRequest,
    StepRequest,
    StepResponse,
    TaskInfo,
)
from baseline.run_agent import run_evaluation
from env import SafePIIEnvironment, TaskType
from env.environment import TASK_CONFIG
from env.state import Observation
from graders import grade_easy, grade_hard, grade_medium


router = APIRouter()
ENV = SafePIIEnvironment()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/reset")
def reset(payload: ResetRequest | None = None) -> Observation:
    task_type = payload.task_type if payload is not None else TaskType.EASY
    return ENV.reset(task_type)


@router.post("/step", response_model=StepResponse)
def step(payload: StepRequest) -> StepResponse:
    observation, reward, done, info = ENV.step(payload.action)
    return StepResponse(observation=observation, reward={"value": reward}, done=done, info=info)


@router.get("/state")
def state() -> Observation:
    return ENV.state()


@router.get("/tasks", response_model=list[TaskInfo])
def tasks() -> list[TaskInfo]:
    output = []
    for task_type, cfg in TASK_CONFIG.items():
        output.append(
            TaskInfo(
                name=task_type,
                allowed_actions=sorted(list(cfg["allowed_actions"]), key=lambda x: x.value),
                max_steps=cfg["max_steps"],
            )
        )
    return output


@router.post("/grader", response_model=GraderResponse)
def grader(payload: GraderRequest) -> GraderResponse:
    if ENV.state().task_type != payload.task_type:
        return GraderResponse(score=0.0)
    if payload.task_type == TaskType.EASY:
        score = grade_easy(payload.pred_entities, ENV.gold_entities)
    elif payload.task_type == TaskType.MEDIUM:
        score = grade_medium(
            payload.pred_entities,
            ENV.gold_entities,
            payload.pred_risk or ENV.state().risk_level.value,
            ENV.gold_risk,
        )
    else:
        score = grade_hard(
            payload.pred_entities,
            ENV.gold_entities,
            payload.pred_risk or ENV.state().risk_level.value,
            ENV.gold_risk,
            ENV.state().document_text,
            payload.redacted_text or ENV.last_redacted_text or ENV.state().document_text,
            payload.steps_used or ENV.state().step_count,
        )
    return GraderResponse(score=score)


@router.post("/baseline", response_model=BaselineResponse)
def baseline(payload: BaselineRequest | None = None) -> BaselineResponse:
    requested = payload.use_ai if payload is not None else True
    return BaselineResponse(scores=run_evaluation(use_ai=requested))
