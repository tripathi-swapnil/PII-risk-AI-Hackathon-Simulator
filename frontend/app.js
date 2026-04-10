const taskSelect = document.getElementById("taskSelect");
const confidenceInput = document.getElementById("confidenceInput");
const classificationInput = document.getElementById("classificationInput");
const redactedTextInput = document.getElementById("redactedTextInput");

const observationView = document.getElementById("observationView");
const tasksView = document.getElementById("tasksView");
const gradeView = document.getElementById("gradeView");
const baselineView = document.getElementById("baselineView");
const logView = document.getElementById("logView");

const resetBtn = document.getElementById("resetBtn");
const stateBtn = document.getElementById("stateBtn");
const gradeBtn = document.getElementById("gradeBtn");
const baselineBtn = document.getElementById("baselineBtn");
const actionButtons = document.querySelectorAll(".action-btn");

let currentObservation = {};
let taskCatalog = [];

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function logEvent(kind, payload) {
  const row = document.createElement("div");
  row.className = "log-line";
  row.textContent = `[${new Date().toLocaleTimeString()}] ${kind}: ${typeof payload === "string" ? payload : JSON.stringify(payload)}`;
  logView.prepend(row);
}

async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const text = await response.text();
  let parsed = {};
  try {
    parsed = text ? JSON.parse(text) : {};
  } catch {
    parsed = { raw: text };
  }

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} ${JSON.stringify(parsed)}`);
  }

  return parsed;
}

function currentTask() {
  return taskSelect.value || "easy";
}

function renderObservation(obs) {
  currentObservation = obs || {};
  observationView.textContent = pretty(currentObservation);
}

async function loadTasks() {
  taskCatalog = await apiRequest("/tasks", { method: "GET" });
  taskSelect.innerHTML = "";
  taskCatalog.forEach((task) => {
    const option = document.createElement("option");
    option.value = task.name;
    option.textContent = `${task.name} (max ${task.max_steps})`;
    taskSelect.appendChild(option);
  });
  tasksView.textContent = pretty(taskCatalog);
}

async function doReset() {
  const payload = { task_type: currentTask() };
  const obs = await apiRequest("/reset", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  renderObservation(obs);
  logEvent("reset", payload);
}

async function doState() {
  const obs = await apiRequest("/state", { method: "GET" });
  renderObservation(obs);
  logEvent("state", { ok: true });
}

function buildAction(actionType) {
  const confidence = Number(confidenceInput.value);
  const action = {
    action_type: actionType,
    confidence: Number.isFinite(confidence) ? confidence : 0.8,
  };

  if (actionType === "classify" && classificationInput.value) {
    action.classification = classificationInput.value;
  }

  if (actionType === "redact" && redactedTextInput.value.trim()) {
    action.redacted_text = redactedTextInput.value.trim();
  }

  return action;
}

async function doStep(actionType) {
  const action = buildAction(actionType);
  const result = await apiRequest("/step", {
    method: "POST",
    body: JSON.stringify({ action }),
  });

  renderObservation(result.observation || {});
  logEvent("step", {
    action_type: actionType,
    reward: result.reward,
    done: result.done,
    info: result.info,
  });
}

async function doGrade() {
  const payload = {
    task_type: currentTask(),
    pred_entities: currentObservation.detected_entities || [],
    pred_risk: currentObservation.risk_level || null,
    redacted_text: currentObservation.document_text || null,
    steps_used: currentObservation.step_count || 0,
  };
  const score = await apiRequest("/grader", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  gradeView.textContent = pretty(score);
  logEvent("grade", score);
}

async function doBaseline() {
  const payload = { use_ai: false };
  const scores = await apiRequest("/baseline", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  baselineView.textContent = pretty(scores);
  logEvent("baseline", scores);
}

function wireEvents() {
  resetBtn.addEventListener("click", () => {
    doReset().catch((err) => logEvent("reset_error", String(err)));
  });

  stateBtn.addEventListener("click", () => {
    doState().catch((err) => logEvent("state_error", String(err)));
  });

  gradeBtn.addEventListener("click", () => {
    doGrade().catch((err) => logEvent("grade_error", String(err)));
  });

  baselineBtn.addEventListener("click", () => {
    doBaseline().catch((err) => logEvent("baseline_error", String(err)));
  });

  actionButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const actionType = btn.dataset.action;
      doStep(actionType).catch((err) => logEvent("step_error", String(err)));
    });
  });
}

async function bootstrap() {
  wireEvents();
  await loadTasks();
  await doReset();
}

bootstrap().catch((err) => {
  logEvent("bootstrap_error", String(err));
});
