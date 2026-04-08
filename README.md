# SafePII-RL

Risk-Calibrated PII Compliance Environment for OpenEnv Round 1.

This project simulates a real-world compliance workflow where an agent must detect, classify, and redact PII from documents under safety constraints.

## Why this environment

- Real-world utility: mirrors workflows in GDPR/HIPAA/CCPA compliance pipelines.
- Safety-first design: missed critical identifiers (SSN/AADHAAR/BANK_ACCOUNT) are heavily penalized.
- Multi-step decision making: detect -> classify -> redact -> finalize/escalate.

## Observation Space

`Observation` fields:

- `document_text: str`
- `detected_entities: List[Entity]`
- `risk_level: low | medium | high`
- `task_type: easy | medium | hard`
- `step_count: int`
- `feedback: str`
- `done: bool`
- `constraint_violated: bool`

## Action Space

`Action` fields:

- `action_type: detect | classify | redact | escalate | finalize`
- `spans: Optional[List[{start, end, label}]]`
- `classification: Optional[str]`
- `redacted_text: Optional[str]`
- `confidence: float (0.0-1.0)`
- `reasoning: Optional[str]`

## Reward Model

- Typed reward model: `Reward(value: float)`.
- Step reward is risk-calibrated and includes partial progress signals.
- Invalid/unsafe behavior is penalized; critical leakage can terminate episodes.

## Tasks

- `easy`: basic PII detection (`detect`, `finalize`, max_steps=5)
- `medium`: detection + risk classification (`detect`, `classify`, `finalize`, max_steps=8)
- `hard`: full workflow (`detect`, `classify`, `redact`, `escalate`, `finalize`, max_steps=15)

## Graders

- `easy_grader`: detection score
- `medium_grader`: detection + classification + critical coverage
- `hard_grader`: detection + classification + redaction + efficiency, with critical-miss cap

All grader scores are clamped in `[0.0, 1.0]` and deterministic.

## API Endpoints

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `POST /baseline`

## Local Setup

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Docker / Hugging Face

```bash
docker build -t safepii-rl .
docker run -p 7860:7860 safepii-rl
```

HF Space should use Docker SDK and expose port `7860`.

## Baseline and Inference

### API baseline endpoint

- `POST /baseline` with `{"use_ai": true}` for OpenAI-backed run.
- Uses `OPENAI_API_KEY` and optional `OPENAI_MODEL` in server environment.

### Current baseline scores

Deterministic rule-based run (`use_ai=false`) currently returns:

| Task | Score |
| --- | --- |
| easy | 0.0000 |
| medium | 0.6667 |
| hard | 0.8233 |

### Mandatory Round 1 script

Root script: `inference.py`

Required env vars:

- `API_BASE_URL` (LLM API endpoint)
- `MODEL_NAME` (LLM model)
- `HF_TOKEN` (HF/API token used by OpenAI client in inference)
- Optional local fallback: `OPENAI_API_KEY`
- Optional: `LOCAL_IMAGE_NAME` if using docker-image-based environment execution
- Optional: `ENV_BASE_URL` (environment API URL, default `http://localhost:7860`)

The script emits structured logs with `[START]`, `[STEP]`, `[END]`.

## Validation

```bash
openenv validate
```

If missing:

```bash
pip install openenv-core
```

## Tests

```bash
python -m unittest discover -s tests -v
```
