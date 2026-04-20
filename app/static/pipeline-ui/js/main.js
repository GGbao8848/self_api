/**
 * Pipeline UI 主入口。
 *
 * 功能：
 *  - 启动 LangGraph pipeline run（可带 class_name_map / full_access / step_gates）
 *  - 本地 localStorage 记录 run_id 列表
 *  - 轮询 run 状态，interrupted=true 时展示 pending_review，支持 confirm/abort
 */

import { loadJson, saveJson } from "/static/shared/js/jsonStorage.js";

const STORAGE_KEY = "self_api_pipeline_runs";
const POLL_INTERVAL_MS = 3000;
const API_BASE = `${location.origin}/api/v1/pipeline`;

const qs = (id) => document.getElementById(id);

const el = {
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
  
  newRunBtn: qs("newRunBtn"),
  tabLocal: qs("tabLocal"),
  tabRemote: qs("tabRemote"),
  settingsBtn: qs("settingsBtn"),
  settingsModal: qs("settingsModal"),
  closeSettingsBtn: qs("closeSettingsBtn"),
  saveSettingsBtn: qs("saveSettingsBtn"),
  settingLocalPath: qs("settingLocalPath"),
  settingRemotePath: qs("settingRemotePath"),

  launchView: qs("launchView"),
  detailView: qs("detailView"),
  headerTitle: qs("headerTitle"),
  headerActions: qs("headerActions"),
  
  detailRunId: qs("detailRunId"),
  detailStatusPill: qs("detailStatusPill"),
  detailRefreshBtn: qs("detailRefreshBtn"),
  detailCopyBtn: qs("detailCopyBtn"),
  detailAbortBtn: qs("detailAbortBtn"),
  stepList: qs("stepList"),
  reviewCard: qs("reviewCard"),
  bottomActionArea: qs("bottomActionArea"),
  noReviewNote: qs("noReviewNote"),
  reviewStep: qs("reviewStep"),
  reviewHint: qs("reviewHint"),
  reviewFacts: qs("reviewFacts"),
  reviewData: qs("reviewData"),
  reviewOverrideForm: qs("reviewOverrideForm"),
  reviewOverride: qs("reviewOverride"),
  confirmBtn: qs("confirmBtn"),
  abortStepBtn: qs("abortStepBtn"),
  reviewStatus: qs("reviewStatus"),
};

let state = {
  runs: loadJson(STORAGE_KEY, []),
  activeRunId: null,
  currentRunData: null,
  pollTimer: null,
  eventSource: null,
  activeTab: "local", // 'local' or 'remote'
  settings: loadJson("pipeline_settings", {
    localPath: "",
    remotePath: ""
  }),
};

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
      execution_mode: payload.execution_mode || "local",
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

function prettyJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function pickFirstObject(source, keys) {
  if (!source || typeof source !== "object") return undefined;
  for (const key of keys) {
    const value = source[key];
    if (value !== undefined) return value;
  }
  return undefined;
}

function extractStepIo(stepData) {
  const input = pickFirstObject(stepData, [
    "input",
    "inputs",
    "request",
    "req",
    "payload",
    "params",
    "params_override",
    "override",
  ]);
  const output = pickFirstObject(stepData, [
    "output",
    "outputs",
    "response",
    "resp",
    "result",
    "results",
  ]);
  return { input, output };
}

function toShortText(value) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return `共 ${value.length} 项`;
  if (typeof value === "object") return `对象(${Object.keys(value).length})`;
  return String(value);
}

function pickReadableFacts(source, limit = 6) {
  if (!source || typeof source !== "object") return [];
  const pairs = [];
  for (const [key, value] of Object.entries(source)) {
    if (pairs.length >= limit) break;
    if (["hint", "note", "instructions"].includes(key)) continue;
    if (value === null || value === undefined || value === "") continue;
    if (Array.isArray(value) && value.length > 8) {
      pairs.push([key, `共 ${value.length} 项`]);
      continue;
    }
    pairs.push([key, toShortText(value)]);
  }
  return pairs;
}

function renderFacts(container, facts) {
  container.innerHTML = "";
  if (!facts.length) {
    const note = document.createElement("div");
    note.className = "agent-note";
    note.textContent = "暂无关键字段，展开高级信息可查看完整数据。";
    container.appendChild(note);
    return;
  }
  container.className = "kv-list";
  for (const [label, value] of facts) {
    const item = document.createElement("div");
    item.className = "kv-item";
    const k = document.createElement("div");
    k.className = "kv-label";
    k.textContent = label;
    const v = document.createElement("div");
    v.className = "kv-value";
    v.textContent = value;
    item.append(k, v);
    container.appendChild(item);
  }
}

function parseMaybeJson(value, fallback = undefined) {
  const text = String(value || "").trim();
  if (!text) return fallback;
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

function createFormField(label, key, value = "", options = {}) {
  const wrap = document.createElement("div");
  wrap.className = options.full ? "review-form-field review-form-field--full" : "review-form-field";
  const lb = document.createElement("label");
  lb.textContent = label;
  const input = options.multiline ? document.createElement("textarea") : document.createElement("input");
  if (!options.multiline) input.type = options.type || "text";
  input.dataset.overrideKey = key;
  input.value = value ?? "";
  if (options.placeholder) input.placeholder = options.placeholder;
  if (options.rows && input.tagName === "TEXTAREA") input.rows = options.rows;
  wrap.append(lb, input);
  return wrap;
}

function renderOverrideForm(stepName, review = {}) {
  el.reviewOverrideForm.innerHTML = "";
  const formRoot = document.createElement("div");
  const title = document.createElement("h4");
  title.textContent = "快捷参数修改（会作为 params_override 提交）";
  formRoot.appendChild(title);
  const grid = document.createElement("div");
  grid.className = "review-form-grid";

  if (stepName === "discover_classes") {
    grid.append(
      createFormField("class_name_map（JSON 对象）", "class_name_map", "{}", {
        multiline: true,
        rows: 3,
        full: true,
        placeholder: '{"louyou1":"louyou","louyou2":"louyou"}',
      }),
      createFormField("final_classes（JSON 数组）", "final_classes", "[]", {
        multiline: true,
        rows: 2,
        full: true,
        placeholder: '["louyou"]',
      }),
    );
  } else if (stepName === "publish_transfer") {
    grid.append(
      createFormField("publish_input_dir", "input_dir", review.publish_input_dir || ""),
      createFormField("execution_mode", "publish_mode", review.execution_mode || "local"),
      createFormField("project_root_dir", "project_root_dir", review.project_root_dir || ""),
      createFormField("detector_name", "detector_name", review.detector_name || ""),
    );
  } else if (stepName === "train") {
    grid.append(
      createFormField("yaml_path", "yaml_path", review.yaml_path || "", { full: true }),
      createFormField("project", "project", review.project || "", { full: true }),
      createFormField("name", "name", review.name || ""),
      createFormField("model", "model", review.model || ""),
      createFormField("epochs", "epochs", String(review.epochs ?? 100), { type: "number" }),
      createFormField("imgsz", "imgsz", String(review.imgsz ?? 640), { type: "number" }),
      createFormField("yolo_train_env", "yolo_train_env", review.yolo_train_env || ""),
      createFormField("project_root_dir", "project_root_dir", "", { full: true }),
    );
  } else {
    grid.append(createFormField("备注（可选）", "_note", "", {
      multiline: true,
      rows: 2,
      full: true,
      placeholder: "该审核点暂无标准可改字段，可直接编辑下方 JSON。",
    }));
  }

  formRoot.appendChild(grid);
  el.reviewOverrideForm.appendChild(formRoot);
  el.reviewOverrideForm.hidden = false;
}

function collectOverrideFormValues() {
  const fields = el.reviewOverrideForm.querySelectorAll("[data-override-key]");
  const override = {};
  fields.forEach((input) => {
    const key = input.dataset.overrideKey;
    const raw = String(input.value || "").trim();
    if (!raw || key === "_note") return;
    if (key === "epochs" || key === "imgsz") {
      override[key] = Number(raw);
      return;
    }
    if (key === "class_name_map") {
      const parsed = parseMaybeJson(raw);
      override[key] = parsed ?? raw;
      return;
    }
    if (key === "final_classes") {
      const parsed = parseMaybeJson(raw);
      override[key] = parsed ?? raw;
      return;
    }
    override[key] = raw;
  });
  return override;
}

function renderRunList() {
  el.runList.innerHTML = "";
  
  el.tabLocal.classList.toggle("active", state.activeTab === "local");
  el.tabRemote.classList.toggle("active", state.activeTab === "remote");
  
  const filteredRuns = state.runs.filter(run => {
    const mode = run.execution_mode || "local";
    if (state.activeTab === "local") {
      return mode === "local";
    } else {
      return mode === "remote_sftp" || mode === "remote_slurm";
    }
  });

  if (filteredRuns.length === 0) {
    const empty = document.createElement("div");
    empty.className = "inline-note text-center";
    empty.style.marginTop = "20px";
    empty.textContent = "暂无记录";
    el.runList.appendChild(empty);
    return;
  }

  for (const run of filteredRuns) {
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

function showLaunchView() {
  state.activeRunId = null;
  stopPolling();
  renderRunList();
  el.headerTitle.textContent = "新建 Run";
  el.headerActions.hidden = true;
  el.launchView.hidden = false;
  el.detailView.hidden = true;
  el.bottomActionArea.hidden = true;
  el.statusBox.hidden = true;

  // Auto-fill projectRootDir based on active tab
  if (state.activeTab === "local" && state.settings.localPath) {
    el.projectRootDir.value = state.settings.localPath;
    el.executionMode.value = "local";
  } else if (state.activeTab === "remote" && state.settings.remotePath) {
    el.projectRootDir.value = state.settings.remotePath;
    el.executionMode.value = "remote_sftp"; // Default remote mode
  }
}

async function openRun(runId, prefetched = null) {
  state.activeRunId = runId;
  renderRunList();
  
  el.headerTitle.textContent = `Run: ${runId.slice(0, 8)}...`;
  el.headerActions.hidden = false;
  el.launchView.hidden = true;
  el.detailView.hidden = false;
  
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
  state.currentRunData = data;
  const statusText = summarizeStatus(data);
  el.detailStatusPill.textContent = statusText;
  el.detailStatusPill.className = `pill ${statusText}`;

  el.stepList.innerHTML = "";
  const allSteps = [
    "healthcheck", "discover_classes", "xml_to_yolo", "review_labels",
    "split_dataset", "crop_augment", "publish_transfer",
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

    const details = document.createElement("details");
    details.className = "step-card";
    if (
      (data.interrupted && data.current_step === stepName) || 
      (data.current_step === stepName && !result) ||
      (result && result.status === "failed")
    ) {
      details.open = true;
    }

    const summaryLine = document.createElement("summary");
    summaryLine.className = "step-summary";
    const header = document.createElement("div");
    header.className = "step-header";
    const name = document.createElement("span");
    name.className = "step__name";
    name.textContent = stepName;
    const pill = document.createElement("span");
    pill.className = `pill ${statusLabel}`;
    pill.textContent = statusLabel;
    header.append(name, pill);

    const summaryText = document.createElement("div");
    summaryText.className = "step__summary";
    summaryText.textContent = result?.summary || "点击展开查看该步骤详情、输入与输出。";
    summaryLine.append(header, summaryText);

    const detailData = result?.data || {};
    const detailBody = document.createElement("div");
    detailBody.className = "step-detail";
    const factsWrap = document.createElement("div");
    renderFacts(factsWrap, pickReadableFacts(detailData, 8));
    detailBody.appendChild(factsWrap);

    const rawDetails = document.createElement("details");
    rawDetails.className = "disclosure";
    const rawSummary = document.createElement("summary");
    rawSummary.textContent = "高级信息：输入/输出与原始 JSON";
    const { input, output } = extractStepIo(detailData);
    const rawPre = document.createElement("pre");
    rawPre.className = "step-detail-json";
    rawPre.textContent = prettyJson({
      status: result?.status || statusLabel,
      summary: result?.summary || "",
      input,
      output,
      data: detailData,
    });
    rawDetails.append(rawSummary, rawPre);
    detailBody.appendChild(rawDetails);

    details.append(summaryLine, detailBody);

    li.appendChild(details);
    el.stepList.appendChild(li);
  }

  if (data.interrupted && data.pending_review) {
    el.reviewCard.hidden = false;
    el.noReviewNote.hidden = true;
    el.bottomActionArea.hidden = false;
    const pending = data.pending_review;
    const stepName = pending.step || data.current_step || "?";
    el.reviewStep.textContent = pending.step || data.current_step || "?";
    el.reviewHint.textContent = pending.instructions || "";
    const dataBlock = { ...(pending.review || {}) };
    el.reviewFacts.className = "review-facts";
    renderFacts(el.reviewFacts, pickReadableFacts(dataBlock, 8));
    el.reviewData.textContent = JSON.stringify(dataBlock, null, 2);
    renderOverrideForm(stepName, dataBlock);
  } else {
    el.reviewCard.hidden = true;
    el.bottomActionArea.hidden = true;
    el.reviewOverrideForm.hidden = true;
    el.reviewOverrideForm.innerHTML = "";
    el.reviewFacts.innerHTML = "";
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
  if (typeof EventSource !== "undefined") {
    try {
      const es = new EventSource(`${API_BASE}/${runId}/events?poll_interval=1`);
      state.eventSource = es;
      es.addEventListener("snapshot", (ev) => {
        if (runId !== state.activeRunId) return;
        try {
          renderDetail(JSON.parse(ev.data));
        } catch { /* ignore */ }
      });
      es.addEventListener("end", () => stopPolling());
      es.onerror = () => {
        stopPolling();
        fallbackPolling(runId);
      };
      return;
    } catch {
      /* EventSource 不可用时退回轮询 */
    }
  }
  fallbackPolling(runId);
}

function fallbackPolling(runId) {
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
  if (state.eventSource) {
    try { state.eventSource.close(); } catch { /* ignore */ }
    state.eventSource = null;
  }
}

async function confirmStep() {
  if (!state.activeRunId) return;
  let override = collectOverrideFormValues();
  const txt = el.reviewOverride.value.trim();
  if (txt) {
    try {
      const jsonOverride = JSON.parse(txt);
      override = { ...override, ...jsonOverride };
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
  showLaunchView();
}

function copyRunParams() {
  const data = state.currentRunData;
  if (!data || !data.initial_params) return;

  const p = data.initial_params;
  if (p.original_dataset) el.originalDataset.value = p.original_dataset;
  if (p.detector_name) el.detectorName.value = p.detector_name;
  if (p.project_root_dir) el.projectRootDir.value = p.project_root_dir;
  if (p.execution_mode) el.executionMode.value = p.execution_mode;
  if (p.yolo_train_env) el.trainEnv.value = p.yolo_train_env;
  if (p.yolo_train_model) el.trainModel.value = p.yolo_train_model;
  if (p.yolo_train_epochs) el.trainEpochs.value = p.yolo_train_epochs;
  if (p.yolo_train_imgsz) el.trainImgsz.value = p.yolo_train_imgsz;
  if (typeof p.full_access === "boolean") el.fullAccess.checked = p.full_access;

  if (p.class_name_map) {
    el.classNameMap.value = JSON.stringify(p.class_name_map, null, 2);
  }
  if (p.final_classes) {
    el.finalClasses.value = JSON.stringify(p.final_classes);
  }

  showLaunchView();
  setStatus(el.statusBox, "已将该 run 的参数填入表单，可修改后重新启动。", "info");
  el.statusBox.hidden = false;
}

el.newRunBtn.addEventListener("click", showLaunchView);
el.tabLocal.addEventListener("click", () => {
  state.activeTab = "local";
  renderRunList();
  showLaunchView();
});
el.tabRemote.addEventListener("click", () => {
  state.activeTab = "remote";
  renderRunList();
  showLaunchView();
});

el.settingsBtn.addEventListener("click", () => {
  el.settingLocalPath.value = state.settings.localPath || "";
  el.settingRemotePath.value = state.settings.remotePath || "";
  el.settingsModal.hidden = false;
});
el.closeSettingsBtn.addEventListener("click", () => {
  el.settingsModal.hidden = true;
});
el.saveSettingsBtn.addEventListener("click", () => {
  state.settings.localPath = el.settingLocalPath.value.trim();
  state.settings.remotePath = el.settingRemotePath.value.trim();
  saveJson("pipeline_settings", state.settings);
  el.settingsModal.hidden = true;
  showLaunchView(); // Update form if needed
});

el.runBtn.addEventListener("click", startRun);
el.previewBtn.addEventListener("click", previewRun);
el.sopSelect.addEventListener("change", onSopChange);
el.refreshAllBtn.addEventListener("click", refreshAll);
el.clearRunsBtn.addEventListener("click", clearRuns);
el.detailRefreshBtn.addEventListener("click", () => {
  if (state.activeRunId) openRun(state.activeRunId);
});
el.detailCopyBtn.addEventListener("click", copyRunParams);
el.detailAbortBtn.addEventListener("click", abortCurrentStep);
el.confirmBtn.addEventListener("click", confirmStep);
el.abortStepBtn.addEventListener("click", abortCurrentStep);

renderRunList();
loadSops();
if (state.runs[0]) {
  openRun(state.runs[0].run_id);
} else {
  showLaunchView();
}
