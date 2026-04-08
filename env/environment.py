import random
from typing import Any

from env.action import Action, ActionType
from env.reward import compute_reward
from env.risk_engine import (
    compute_document_risk,
    constraint_violated,
    count_missed_critical,
)
from env.state import Entity, Observation, RiskLevel, TaskType
from tasks import load_easy_tasks, load_hard_tasks, load_medium_tasks


TASK_CONFIG: dict[TaskType, dict[str, Any]] = {
    TaskType.EASY: {"allowed_actions": {ActionType.DETECT, ActionType.FINALIZE}, "max_steps": 5},
    TaskType.MEDIUM: {
        "allowed_actions": {ActionType.DETECT, ActionType.CLASSIFY, ActionType.FINALIZE},
        "max_steps": 8,
    },
    TaskType.HARD: {
        "allowed_actions": {
            ActionType.DETECT,
            ActionType.CLASSIFY,
            ActionType.REDACT,
            ActionType.ESCALATE,
            ActionType.FINALIZE,
        },
        "max_steps": 15,
    },
}


class SafePIIEnvironment:
    def __init__(self) -> None:
        self._rng = random.Random(42)
        self._current_task_type = TaskType.EASY
        self._observation: Observation | None = None
        self._gold_entities: list[Entity] = []
        self._gold_risk = RiskLevel.LOW
        self._last_redacted_text: str | None = None

    def _load_tasks(self, task_type: TaskType) -> list[dict[str, Any]]:
        if task_type == TaskType.EASY:
            return load_easy_tasks()
        if task_type == TaskType.MEDIUM:
            return load_medium_tasks()
        return load_hard_tasks()

    def _materialize_entities(self, text: str, entities: list[dict[str, str]]) -> list[Entity]:
        materialized: list[Entity] = []
        for ent in entities:
            value = ent["text"]
            start = text.find(value)
            if start == -1:
                continue
            materialized.append(
                Entity(
                    text=value,
                    label=ent["label"],
                    start=start,
                    end=start + len(value),
                    confidence=1.0,
                )
            )
        return materialized

    def reset(self, task_type: TaskType) -> Observation:
        self._current_task_type = task_type
        tasks = self._load_tasks(task_type)
        sample = self._rng.choice(tasks)
        text = sample["text"]

        self._gold_entities = self._materialize_entities(text, sample["entities"])
        self._gold_risk = compute_document_risk(self._gold_entities)
        self._last_redacted_text = None

        self._observation = Observation(
            document_text=text,
            detected_entities=[],
            risk_level=RiskLevel.LOW,
            task_type=task_type,
            step_count=0,
            feedback="Episode reset.",
            done=False,
            constraint_violated=False,
        )
        return self._observation

    def state(self) -> Observation:
        if self._observation is None:
            return self.reset(TaskType.EASY)
        return self._observation

    def _score_detection(self, detected: list[Entity]) -> float:
        gold_spans = {(e.start, e.end, e.label) for e in self._gold_entities}
        tp = sum(1 for e in detected if (e.start, e.end, e.label) in gold_spans)
        fp = max(0, len(detected) - tp)
        return (tp * 0.2) - (fp * 0.15)

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        obs = self.state()
        if obs.done:
            return obs, 0.0, True, {"message": "Episode already done."}

        cfg = TASK_CONFIG[obs.task_type]
        obs.step_count += 1
        base_reward = -0.05
        info: dict[str, Any] = {"task_type": obs.task_type.value}

        if action.action_type not in cfg["allowed_actions"]:
            obs.constraint_violated = True
            obs.feedback = f"Action {action.action_type.value} not allowed for {obs.task_type.value}."
            reward = -1.0
            obs.done = True
            return obs, reward, True, info

        if action.action_type == ActionType.DETECT:
            detected = []
            for span in action.spans or []:
                detected.append(
                    Entity(
                        text=obs.document_text[span.start:span.end],
                        label=span.label,
                        start=span.start,
                        end=span.end,
                        confidence=action.confidence,
                    )
                )
            obs.detected_entities = detected
            obs.risk_level = compute_document_risk(detected)
            base_reward = self._score_detection(detected)
            obs.feedback = "Detection updated."

        elif action.action_type == ActionType.CLASSIFY:
            classified = (action.classification or "").lower().strip()
            base_reward = 0.2 if classified == self._gold_risk.value else -0.2
            obs.risk_level = RiskLevel(classified) if classified in RiskLevel._value2member_map_ else obs.risk_level
            obs.feedback = "Classification processed."

        elif action.action_type == ActionType.REDACT:
            redacted = action.redacted_text or obs.document_text
            self._last_redacted_text = redacted
            violated = constraint_violated(redacted, self._gold_entities)
            if violated:
                obs.constraint_violated = True
                obs.feedback = "Critical leakage detected in redaction."
                obs.done = True
                return obs, -1.0, True, {"constraint": "critical_leakage"}
            base_reward = 0.4
            obs.feedback = "Redaction accepted."
            obs.document_text = redacted

        elif action.action_type == ActionType.ESCALATE:
            base_reward = 0.15 if action.confidence < 0.4 else -0.05
            obs.feedback = "Escalation handled."

        elif action.action_type == ActionType.FINALIZE:
            obs.done = True
            base_reward = 0.1
            obs.feedback = "Episode finalized."

        missed_critical = count_missed_critical(obs.detected_entities, self._gold_entities)
        reward = compute_reward(action.action_type, base_reward, self._gold_entities, missed_critical)

        if obs.step_count >= cfg["max_steps"]:
            obs.done = True
            obs.feedback = "Max steps reached."

        info.update(
            {
                "gold_risk": self._gold_risk.value,
                "missed_critical": missed_critical,
                "max_steps": cfg["max_steps"],
            }
        )
        return obs, reward, obs.done, info

    @property
    def gold_entities(self) -> list[Entity]:
        return self._gold_entities

    @property
    def gold_risk(self) -> RiskLevel:
        return self._gold_risk

    @property
    def last_redacted_text(self) -> str | None:
        return self._last_redacted_text
