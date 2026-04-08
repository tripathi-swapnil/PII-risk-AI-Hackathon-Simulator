import json
import os
from typing import Any

from dotenv import load_dotenv

from agent.pii_policy import build_rule_based_action
from env import Action, ActionType, SafePIIEnvironment, Span, TaskType
from graders import grade_easy, grade_hard, grade_medium

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


load_dotenv()


def _action_from_dict(data: dict[str, Any], task_type: TaskType, step_idx: int, document_text: str) -> Action:
    fallback = build_rule_based_action(task_type, step_idx, document_text)
    action_type = data.get("action_type") or fallback["action_type"]
    confidence = float(data.get("confidence", fallback.get("confidence", 0.5)))
    confidence = max(0.0, min(1.0, confidence))

    spans_payload = data.get("spans") if isinstance(data.get("spans"), list) else fallback.get("spans")
    spans = None
    if spans_payload:
        spans = [Span(**span) for span in spans_payload]

    classification = data.get("classification", fallback.get("classification"))
    redacted_text = data.get("redacted_text", fallback.get("redacted_text"))
    reasoning = data.get("reasoning", fallback.get("reasoning"))

    return Action(
        action_type=ActionType(action_type),
        spans=spans,
        classification=classification,
        redacted_text=redacted_text,
        confidence=confidence,
        reasoning=reasoning,
    )


def _rule_based_action(env: SafePIIEnvironment, task_type: TaskType, step_idx: int) -> Action:
    policy_action = build_rule_based_action(task_type, step_idx, env.state().document_text)
    return _action_from_dict(policy_action, task_type, step_idx, env.state().document_text)


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    payload = raw[start : end + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}


def _ai_action(client: Any, env: SafePIIEnvironment, task_type: TaskType, step_idx: int) -> Action:
    obs = env.state()
    allowed = {
        TaskType.EASY: ["detect", "finalize"],
        TaskType.MEDIUM: ["detect", "classify", "finalize"],
        TaskType.HARD: ["detect", "classify", "redact", "escalate", "finalize"],
    }[task_type]
    prompt = (
        "You are a PII compliance agent. Return ONLY valid JSON with keys: "
        "action_type, confidence, spans(optional), classification(optional), redacted_text(optional), reasoning(optional). "
        f"Allowed action_type values now: {allowed}. "
        f"Step: {step_idx}. Document: {obs.document_text}. "
        "For spans, use [{start,end,label}] char indices."
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content or "{}"
    data = _extract_json(content)
    return _action_from_dict(data, task_type, step_idx, obs.document_text)


def run_single_task(task_type: TaskType, use_ai: bool = True) -> float:
    env = SafePIIEnvironment()
    obs = env.reset(task_type)
    client = None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if use_ai and OpenAI and api_key:
        client = OpenAI(api_key=api_key)
    step_idx = 0
    while True:
        if client is not None:
            try:
                action = _ai_action(client, env, task_type, step_idx)
            except Exception:
                action = _rule_based_action(env, task_type, step_idx)
        else:
            action = _rule_based_action(env, task_type, step_idx)

        final_obs, _, done, _ = env.step(action)
        step_idx += 1
        if done or action.action_type == ActionType.FINALIZE:
            break

    if task_type == TaskType.EASY:
        return grade_easy(final_obs.detected_entities, env.gold_entities)
    if task_type == TaskType.MEDIUM:
        return grade_medium(
            final_obs.detected_entities,
            env.gold_entities,
            final_obs.risk_level,
            env.gold_risk,
        )
    return grade_hard(
        final_obs.detected_entities,
        env.gold_entities,
        final_obs.risk_level,
        env.gold_risk,
        obs.document_text,
        env.last_redacted_text or obs.document_text,
        final_obs.step_count,
    )


def run_evaluation(use_ai: bool = True) -> dict[str, float]:
    return {
        "easy": run_single_task(TaskType.EASY, use_ai=use_ai),
        "medium": run_single_task(TaskType.MEDIUM, use_ai=use_ai),
        "hard": run_single_task(TaskType.HARD, use_ai=use_ai),
    }


if __name__ == "__main__":
    print(run_evaluation(use_ai=True))
