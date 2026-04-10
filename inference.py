import json
import os
import re
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> bool:  # type: ignore[override]
        return False


try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

try:
    from agent.pii_policy import build_rule_based_action
except Exception:  # pragma: no cover
    build_rule_based_action = None  # type: ignore[assignment]


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


def _http_post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {raw[:200]}") from exc


def model_action(client: Any, task: str, document_text: str, step: int) -> dict[str, Any]:
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
    if callable(build_rule_based_action):
        return build_rule_based_action(task, step - 1, document_text, detected_entities)

    # Local fallback so inference never crashes if project imports fail in evaluator.
    if step == 1:
        spans: list[dict[str, Any]] = []
        for match in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", document_text):
            spans.append({"start": match.start(), "end": match.end(), "label": "EMAIL"})
        for match in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", document_text):
            spans.append({"start": match.start(), "end": match.end(), "label": "SSN"})
        for match in re.finditer(r"\b\d{4}-\d{4}-\d{4}\b", document_text):
            spans.append({"start": match.start(), "end": match.end(), "label": "AADHAAR"})
        return {
            "action_type": "detect",
            "spans": spans,
            "confidence": 0.7,
            "reasoning": "local_fallback_detection",
        }

    if task in {"medium", "hard"} and step == 2:
        return {
            "action_type": "classify",
            "classification": "medium",
            "confidence": 0.6,
            "reasoning": "local_fallback_classification",
        }

    if task == "hard" and step == 3:
        return {
            "action_type": "redact",
            "redacted_text": document_text,
            "confidence": 0.6,
            "reasoning": "local_fallback_redaction",
        }

    return {
        "action_type": "finalize",
        "confidence": 0.95,
        "reasoning": "local_fallback_finalize",
    }


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


def create_client() -> tuple[Any | None, str | None]:
    if OpenAI is None:
        return None, "openai_client_unavailable"
    try:
        if HF_TOKEN:
            return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN), None
        if OPENAI_API_KEY:
            return OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY), None
        return None, "missing_api_credentials"
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


def run_task(client: Any | None, task: str, client_error: str | None = None) -> float:
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    observation: dict[str, Any] = {}

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
    try:
        observation = _http_post_json(
            f"{ENV_BASE_URL}/reset",
            {"task_type": task},
            timeout=30,
        )

        for step in range(1, MAX_STEPS[task] + 1):
            if observation.get("done"):
                break

            error = client_error
            if client is not None:
                try:
                    action = model_action(client, task, observation.get("document_text", ""), step)
                    action = normalize_action(action, task, step, observation.get("document_text", ""))
                    error = None
                except Exception as exc:
                    error = str(exc)
                    action = rule_based_action(
                        task,
                        step,
                        observation.get("document_text", ""),
                        observation.get("detected_entities", []),
                    )
            else:
                action = rule_based_action(
                    task,
                    step,
                    observation.get("document_text", ""),
                    observation.get("detected_entities", []),
                )

            result = _http_post_json(
                f"{ENV_BASE_URL}/step",
                {"action": action},
                timeout=30,
            )

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
        grade_response = _http_post_json(
            f"{ENV_BASE_URL}/grader",
            grader_payload,
            timeout=30,
        )
        score = float(grade_response.get("score", 0.0))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        score = 0.0
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    client, client_error = create_client()
    for task in TASKS:
        run_task(client, task, client_error=client_error)


if __name__ == "__main__":
    main()
