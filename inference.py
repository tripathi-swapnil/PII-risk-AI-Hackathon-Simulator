import json
import os
from typing import Any

from dotenv import load_dotenv
import requests
from openai import OpenAI

from agent.pii_policy import build_rule_based_action


load_dotenv()

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = os.getenv("BENCHMARK", "safepii-rl")

MAX_STEPS = {"easy": 5, "medium": 8, "hard": 15}
TASKS = ["easy", "medium", "hard"]
SUCCESS_SCORE_THRESHOLD = 0.5


def _bool_text(value: bool) -> str:
    return str(value).lower()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={_bool_text(done)} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool_text(success)} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _extract_json(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}


def model_action(client: OpenAI, task: str, document_text: str, step: int) -> dict[str, Any]:
    prompt = (
        "Return only JSON with keys action_type, confidence, spans(optional), "
        "classification(optional), redacted_text(optional), reasoning(optional). "
        f"Task={task}, Step={step}. Document={document_text}"
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = completion.choices[0].message.content or "{}"
    return _extract_json(content)


def rule_based_action(task: str, step: int, document_text: str, detected_entities: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return build_rule_based_action(task, step - 1, document_text, detected_entities)


def normalize_action(action: dict[str, Any], task: str, step: int, document_text: str) -> dict[str, Any]:
    if "action_type" not in action:
        return rule_based_action(task, step, document_text)
    normalized = {"action_type": action["action_type"], "confidence": float(action.get("confidence", 0.5))}
    if "spans" in action and isinstance(action["spans"], list):
        normalized["spans"] = action["spans"]
    if "classification" in action:
        normalized["classification"] = action["classification"]
    if "redacted_text" in action:
        normalized["redacted_text"] = action["redacted_text"]
    if "reasoning" in action:
        normalized["reasoning"] = action["reasoning"]
    return normalized


def create_client() -> OpenAI:
    if HF_TOKEN:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    if OPENAI_API_KEY:
        return OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
    raise RuntimeError("Missing credentials: set OPENAI_API_KEY for OpenAI or HF_TOKEN for custom endpoint.")


def run_task(client: OpenAI, task: str) -> float:
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    observation: dict[str, Any] = {}

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
    try:
        response = requests.post(f"{ENV_BASE_URL}/reset", json={"task_type": task}, timeout=30)
        response.raise_for_status()
        observation = response.json()

        for step in range(1, MAX_STEPS[task] + 1):
            if observation.get("done"):
                break

            error = None
            try:
                action = model_action(client, task, observation.get("document_text", ""), step)
                action = normalize_action(action, task, step, observation.get("document_text", ""))
            except Exception as exc:
                error = str(exc)
                action = rule_based_action(
                    task,
                    step,
                    observation.get("document_text", ""),
                    observation.get("detected_entities", []),
                )

            step_response = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
            step_response.raise_for_status()
            result = step_response.json()

            observation = result.get("observation", {})
            reward = float(result.get("reward", {}).get("value", 0.0))
            reward = min(max(reward, 0.0), 1.0)
            done = bool(result.get("done", False))
            info = result.get("info", {})
            step_error = info.get("last_action_error")
            error = step_error if step_error else error

            rewards.append(reward)
            steps_taken = step
            action_str = json.dumps(action, separators=(",", ":"), ensure_ascii=True)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        grader_payload = {
            "task_type": task,
            "pred_entities": observation.get("detected_entities", []),
            "pred_risk": observation.get("risk_level"),
            "redacted_text": observation.get("document_text"),
            "steps_used": observation.get("step_count", steps_taken),
        }
        grade_response = requests.post(f"{ENV_BASE_URL}/grader", json=grader_payload, timeout=30)
        grade_response.raise_for_status()
        score = float(grade_response.json().get("score", 0.0))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        score = 0.0
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    client = create_client()
    for task in TASKS:
        run_task(client, task)


if __name__ == "__main__":
    main()
