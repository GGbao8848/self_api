/**
 * Pipeline UI 主入口。
 *
 * 功能：
 *  - 启动 LangGraph pipeline run（可带 class_name_map / full_access / step_gates）
 *  - 本地 localStorage 记录 run_id 列表
 *  - 轮询 run 状态，interrupted=true 时展示 pending_review，支持 confirm/abort
 */

import { loadJson, saveJson } from "/static/train-ui/js/lib/jsonStorage.js";

const STORAGE_KEY = "self_api_pipeline_runs";
const POLL_INTERVAL_MS = 3000;
const API_BASE = `${location.origin}/api/v1/pipeline`;

const qs = (id) => document.getElementById(id);

const el = {
  apiUrlLabel: qs("apiUrlLabel"),
  sopSelect: qs("sopSelect"),
  sopDescription: qs("sopDescription"),
  originalDataset: qs("originalDataset"),
  detectorName: qs("detectorName"),
  projectRootDir: qs("projectRootDir"),
  executionMode: qs("executionMode"),
  trainEnv: qs("trainEnv"),
  trainModel: qs("trainModel"),
  trainEpochs: qs("trainEpochs"),
  trainImgsz: qs("trainImgsz"),
  fullAccess: qs("fullAccess"),
  classNameMap: qs("classNameMap"),
  finalClasses: qs("finalClasses"),
  runBtn: qs("runBtn"),
  previewBtn: qs("previewBtn"),
  statusBox: qs("statusBox"),
  previewText: qs("previewText"),
  runList: qs("runList"),
  refreshAllBtn: qs("refreshAllBtn"),
  clearRunsBtn: qs("clearRunsBtn"),
  detailPanel: qs("detailPanel"),
  detailRunId: qs("detailRunId"),
  detailStatusPill: qs("detailStatusPill"),
  detailRefreshBtn: qs("detailRefreshBtn"),
  detailAbortBtn: qs("detailAbortBtn"),
  stepList: qs("stepList"),
  reviewCard: qs("reviewCard"),
  noReviewNote: qs("noReviewNote"),
  reviewStep: qs("reviewStep"),
  reviewHint: qs("reviewHint"),
  reviewData: qs("reviewData"),
  reviewOverride: qs("reviewOverride"),
  confirmBtn: qs("confirmBtn"),
  abortStepBtn: qs("abortStepBtn"),
  reviewStatus: qs("reviewStatus"),
};

let state = {
  runs: loadJson(STORAGE_KEY, []),
  activeRunId: null,
  pollTimer: null,
};

el.apiUrlLabel.textContent = API_BASE;

let sopCache = [];

async function loadSops() {
  try {
    const resp = await fetchJson(`${API_BASE}/sops`);
    sopCache = resp.sops || [];
    for (const sop of sopCache) {
      const opt = document.createElement("option");
      opt.value = sop.id;
      opt.textContent = `${sop.name} (${sop.id})`;
      el.sopSelect.appendChild(opt);
    }
  } catch (exc) {
    setStatus(el.statusBox, `加载 SOP 失败：${exc.message}`, "warn");
  }
}

function onSopChange() {
  const sopId = el.sopSelect.value;
  if (!sopId) {
    el.sopDescription.textContent = "选择上方 SOP 查看详情。";
    return;
  }
  const sop = sopCache.find((s) => s.id === sopId);
  if (!sop) return;
  el.sopDescription.textContent = sop.description;
  const d = sop.defaults || {};
  if (d.execution_mode) el.executionMode.value = d.execution_mode;
  if (d.yolo_train_model) el.trainModel.value = d.yolo_train_model;
  if (d.yolo_train_epochs) el.trainEpochs.value = d.yolo_train_epochs;
  if (d.yolo_train_imgsz) el.trainImgsz.value = d.yolo_train_imgsz;
  if (typeof d.full_access === "boolean") el.fullAccess.checked = d.full_access;
}


function setStatus(target, text, kind = "info") {
  target.textContent = text;
  target.className = `status ${kind}`;
}

function persistRuns() {
  saveJson(STORAGE_KEY, state.runs);
}

function buildRunPayload() {
  const payload = {
    original_dataset: el.originalDataset.value.trim(),
    detector_name: el.detectorName.value.trim(),
    project_root_dir: el.projectRootDir.value.trim(),
    execution_mode: el.executionMode.value,
    yolo_train_env: el.trainEnv.value.trim(),
    yolo_train_model: el.trainModel.value.trim(),
    yolo_train_epochs: Number(el.trainEpochs.value) || 100,
    yolo_train_imgsz: Number(el.trainImgsz.value) || 640,
    full_access: el.fullAccess.checked,
    self_api_url: location.origin,
  };

  const mapText = el.classNameMap.value.trim();
  if (mapText) {
    payload.class_name_map = JSON.parse(mapText);
  }
  const finalText = el.finalClasses.value.trim();
  if (finalText) {
    payload.final_classes = JSON.parse(finalText);
  }
  return payload;
}

function validatePayload(payload) {
  const required = ["original_dataset", "detector_name", "project_root_dir", "yolo_train_env"];
  for (const key of required) {
    if (!payload[key]) {
      throw new Error(`字段 ${key} 必填`);
    }
  }
}

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    ...options,
  });
  const text = await resp.text();
  let body;
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    body = { detail: text };
  }
  if (!resp.ok) {
    const msg = body?.detail || `HTTP ${resp.status}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return body;
}

async function startRun() {
  let payload;
  try {
    payload = buildRunPayload();
    validatePayload(payload);
  } catch (exc) {
    setStatus(el.statusBox, `参数错误：${exc.message}`, "error");
    return;
  }

  setStatus(el.statusBox, "正在启动 run...", "info");
  el.runBtn.disabled = true;
  const sopId = el.sopSelect.value;
  const runUrl = sopId ? `${API_BASE}/sops/${encodeURIComponent(sopId)}/run` : `${API_BASE}/run`;
  try {
    const resp = await fetchJson(runUrl, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.runs.unshift({
      run_id: resp.run_id,
      detector_name: payload.detector_name,
      started_at: new Date().toISOString(),
      last_status: summarizeStatus(resp),
    });
    persistRuns();
    renderRunList();
    await openRun(resp.run_id, resp);
    setStatus(el.statusBox, `run 已启动：${resp.run_id}`, "success");
  } catch (exc) {
    setStatus(el.statusBox, `启动失败：${exc.message}`, "error");
  } finally {
    el.runBtn.disabled = false;
  }
}

function previewRun() {
  try {
    const payload = buildRunPayload();
    el.previewText.value = JSON.stringify(payload, null, 2);
    el.previewText.closest("details").open = true;
  } catch (exc) {
    setStatus(el.statusBox, `预览失败：${exc.message}`, "error");
  }
}

function summarizeStatus(resp) {
  if (resp.completed) {
    return resp.error ? "failed" : "completed";
  }
  if (resp.interrupted) {
    return "waiting";
  }
  return "running";
}

function renderRunList() {
  el.runList.innerHTML = "";
  for (const run of state.runs) {
    const li = document.createElement("li");
    if (run.run_id === state.activeRunId) li.classList.add("active");
    const detector = document.createElement("span");
    detector.className = "run-list__detector";
    detector.textContent = run.detector_name || "(no detector)";
    const pill = document.createElement("span");
    pill.className = `pill ${run.last_status || ""}`;
    pill.textContent = run.last_status || "unknown";
    const rid = document.createElement("span");
    rid.className = "run-list__id";
    rid.textContent = run.run_id.slice(0, 8);
    li.append(detector, pill, rid);
    li.addEventListener("click", () => openRun(run.run_id));
    el.runList.appendChild(li);
  }
}

async function openRun(runId, prefetched = null) {
  state.activeRunId = runId;
  renderRunList();
  el.detailPanel.hidden = false;
  el.detailRunId.textContent = runId;
  stopPolling();

  const data = prefetched || (await fetchJson(`${API_BASE}/${runId}`).catch((e) => {
    setStatus(el.reviewStatus, `加载失败：${e.message}`, "error");
    return null;
  }));
  if (!data) return;
  renderDetail(data);
  if (!data.completed) {
    startPolling(runId);
  }
}

function renderDetail(data) {
  const statusText = summarizeStatus(data);
  el.detailStatusPill.textContent = statusText;
  el.detailStatusPill.className = `pill ${statusText}`;

  el.stepList.innerHTML = "";
  const allSteps = [
    "healthcheck", "discover_classes", "xml_to_yolo", "review_labels",
    "split_dataset", "crop_augment", "build_yaml", "publish_transfer",
    "train", "poll_train", "review_result",
  ];
  for (const stepName of allSteps) {
    const result = data.step_results?.[stepName];
    const li = document.createElement("li");
    let cls = "";
    let statusLabel = "pending";
    if (result) {
      if (result.status === "ok") { cls = "step--ok"; statusLabel = "ok"; }
      else if (result.status === "failed") { cls = "step--failed"; statusLabel = "failed"; }
      else if (result.status === "skipped") { cls = "step--skipped"; statusLabel = "skipped"; }
    }
    if (data.current_step === stepName && !result) {
      cls = "step--running"; statusLabel = "running";
    }
    if (data.interrupted && data.current_step === stepName) {
      cls = "step--waiting"; statusLabel = "waiting";
    }
    li.className = cls;

    const body = document.createElement("div");
    body.className = "step-body";
    const name = document.createElement("span");
    name.className = "step__name";
    name.textContent = stepName;
    const pill = document.createElement("span");
    pill.className = `pill ${statusLabel}`;
    pill.textContent = statusLabel;
    const summary = document.createElement("div");
    summary.className = "step__summary";
    summary.textContent = result?.summary || "";
    body.append(name, " ", pill);
    if (summary.textContent) body.appendChild(summary);
    li.appendChild(body);
    el.stepList.appendChild(li);
  }

  if (data.interrupted && data.pending_review) {
    el.reviewCard.hidden = false;
    el.noReviewNote.hidden = true;
    const pending = data.pending_review;
    el.reviewStep.textContent = pending.step || data.current_step || "?";
    el.reviewHint.textContent = pending.instructions || "";
    const dataBlock = { ...(pending.review || {}) };
    el.reviewData.textContent = JSON.stringify(dataBlock, null, 2);
  } else {
    el.reviewCard.hidden = true;
    el.noReviewNote.hidden = false;
    if (data.completed) {
      el.noReviewNote.textContent = data.error
        ? `run 已终止：${data.error}`
        : "run 已完成。";
    } else {
      el.noReviewNote.textContent = "当前没有待人工审核的步骤。";
    }
  }

  const runIdx = state.runs.findIndex((r) => r.run_id === data.run_id);
  if (runIdx >= 0) {
    state.runs[runIdx].last_status = statusText;
    persistRuns();
    renderRunList();
  }
}

function startPolling(runId) {
  stopPolling();
  state.pollTimer = setInterval(async () => {
    if (runId !== state.activeRunId) return;
    try {
      const data = await fetchJson(`${API_BASE}/${runId}`);
      renderDetail(data);
      if (data.completed) stopPolling();
    } catch (e) {
      setStatus(el.reviewStatus, `轮询失败：${e.message}`, "error");
    }
  }, POLL_INTERVAL_MS);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function confirmStep() {
  if (!state.activeRunId) return;
  let override = {};
  const txt = el.reviewOverride.value.trim();
  if (txt) {
    try {
      override = JSON.parse(txt);
    } catch {
      setStatus(el.reviewStatus, "params_override 不是合法 JSON", "error");
      return;
    }
  }
  el.confirmBtn.disabled = true;
  setStatus(el.reviewStatus, "提交确认，等待流程推进...", "info");
  try {
    const data = await fetchJson(
      `${API_BASE}/${state.activeRunId}/confirm`,
      {
        method: "POST",
        body: JSON.stringify({ decision: "confirm", params_override: override }),
      },
    );
    renderDetail(data);
    el.reviewOverride.value = "";
    setStatus(el.reviewStatus, "已确认，流程继续。", "success");
    if (!data.completed) startPolling(state.activeRunId);
  } catch (exc) {
    setStatus(el.reviewStatus, `确认失败：${exc.message}`, "error");
  } finally {
    el.confirmBtn.disabled = false;
  }
}

async function abortCurrentStep() {
  if (!state.activeRunId) return;
  if (!confirm("确认要中止当前 run 吗？")) return;
  el.abortStepBtn.disabled = true;
  try {
    const data = await fetchJson(
      `${API_BASE}/${state.activeRunId}/abort`,
      { method: "POST" },
    );
    renderDetail(data);
    setStatus(el.reviewStatus, "run 已中止。", "warn");
    stopPolling();
  } catch (exc) {
    setStatus(el.reviewStatus, `中止失败：${exc.message}`, "error");
  } finally {
    el.abortStepBtn.disabled = false;
  }
}

async function refreshAll() {
  for (const run of state.runs) {
    try {
      const data = await fetchJson(`${API_BASE}/${run.run_id}`);
      run.last_status = summarizeStatus(data);
    } catch {
      run.last_status = "unknown";
    }
  }
  persistRuns();
  renderRunList();
}

function clearRuns() {
  if (!confirm("清空本地 run 记录？（服务端已存在的 run 不会被删除）")) return;
  state.runs = [];
  state.activeRunId = null;
  persistRuns();
  renderRunList();
  el.detailPanel.hidden = true;
  stopPolling();
}

el.runBtn.addEventListener("click", startRun);
el.previewBtn.addEventListener("click", previewRun);
el.sopSelect.addEventListener("change", onSopChange);
el.refreshAllBtn.addEventListener("click", refreshAll);
el.clearRunsBtn.addEventListener("click", clearRuns);
el.detailRefreshBtn.addEventListener("click", () => {
  if (state.activeRunId) openRun(state.activeRunId);
});
el.detailAbortBtn.addEventListener("click", abortCurrentStep);
el.confirmBtn.addEventListener("click", confirmStep);
el.abortStepBtn.addEventListener("click", abortCurrentStep);

renderRunList();
loadSops();
if (state.runs[0]) {
  openRun(state.runs[0].run_id);
}
