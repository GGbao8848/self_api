/**
 * Pipeline UI 主入口。
 *
 * 功能：
 *  - 启动 LangGraph pipeline run（可带 class_name_map / full_access / step_gates）
 *  - 本地 localStorage 记录 run_id 列表
 *  - 轮询 run 状态，interrupted=true 时展示 pending_review，支持 confirm/abort
 */

import { loadJson, saveJson } from "/static/shared/js/jsonStorage.js";
import {
  mountClassMappingForm,
  collectClassMappingPayload,
} from "/static/pipeline-ui/js/classMappingUi.js";

const STORAGE_KEY = "self_api_pipeline_runs";
const POLL_INTERVAL_MS = 3000;
const API_BASE = `${location.origin}/api/v1/pipeline`;

const STEP_META = {
  label_transform_review: {
    label: "标注检查与转换修改",
    goal: "完成类别检查、标签转换和最终标签口径确认。",
    reviewQuestion: "确认类别映射、标签转换和索引定义都正确。",
    reviewAction: "重点看类别列表、class_to_id 和 labels 输出位置。",
    expected: "通过后会进入数据集划分。",
    checklist: [
      "类别合并/改名规则是否正确。",
      "标签转换结果是否完整。",
      "最终类别与索引是否可直接训练。",
    ],
  },
  healthcheck: {
    label: "环境检查",
    goal: "确认 self_api 服务当前可用，整个流程可以正常开始。",
    reviewQuestion: "检查当前服务是否可用，流程能不能继续。",
    reviewAction: "这里只看服务是否连通即可；失败时先修复环境，再重新运行。",
    expected: "成功时会明确告诉你服务可用；失败时流程会直接停下，不建议继续看后面的步骤。",
    checklist: [
      "接口是否可访问。",
      "如果失败，优先检查服务地址、鉴权和后端是否启动。",
    ],
  },
  discover_classes: {
    label: "类别发现",
    goal: "扫描原始 XML，找出数据里实际出现了哪些类别。",
    reviewQuestion: "确认这些类别是不是你想训练的目标类别。",
    reviewAction: "重点看有没有错别字、同义词、脏类别，是否需要合并或改名。",
    expected: "确认后，后续转换会按你的类别映射生成训练标签。",
    checklist: [
      "保留的类别是否都需要训练。",
      "同一类目标是否存在多个写法，需要合并。",
      "最终训练类别顺序或索引是否符合你的预期。",
    ],
  },
  xml_to_yolo: {
    label: "标签转换",
    goal: "把 XML 标注转换成 YOLO 训练所需的标签文件。",
    reviewQuestion: "确认转换是否成功，训练标签有没有正确生成。",
    reviewAction: "优先看成功转换了多少文件、生成了多少框，以及标签目录是否落在预期位置。",
    expected: "成功后会得到可继续划分与训练的 labels 和 classes 信息。",
    checklist: [
      "转换文件数是否接近原始 XML 文件数。",
      "标签目录是否生成在正确位置。",
      "类别列表是否与前一步确认结果一致。",
    ],
  },
  review_labels: {
    label: "标签确认",
    goal: "在真正切分数据前，再确认一次最终训练类别和索引。",
    reviewQuestion: "确认 classes.txt 与标签索引是否就是你要拿去训练的版本。",
    reviewAction: "只看类别名、顺序和索引是否正确；不对就直接改，不需要看原始 JSON。",
    expected: "确认后系统会用这份定义重建 labels/classes.txt，然后继续后续流程。",
    checklist: [
      "类别名是否清晰且没有脏值。",
      "类别顺序或显式索引是否与训练要求一致。",
      "如果做了修改，确认这就是最终训练口径。",
    ],
  },
  split_dataset: {
    label: "数据集划分",
    goal: "把数据划分到 train/val/test，形成可训练的数据结构。",
    reviewQuestion: "确认训练集和验证集是否已经按预期生成。",
    reviewAction: "主要看输出目录和各 split 数量是否合理。",
    expected: "成功后会得到一个新的数据集版本目录，后续步骤会基于它继续处理。",
    checklist: [
      "train/val/test 数量是否合理。",
      "输出目录是否在预期位置。",
    ],
  },
  crop_window: {
    label: "滑窗裁剪",
    goal: "把大图按滑窗策略切分为训练小图。",
    reviewQuestion: "确认滑窗窗口数量与输出路径是否符合预期。",
    reviewAction: "重点看 generated_crops、generated_labels 与 crop 输出目录。",
    expected: "完成后进入增强阶段。",
    checklist: [
      "窗口数量是否符合预期。",
      "裁剪输出目录是否正确。",
    ],
  },
  augment_only: {
    label: "增强",
    goal: "对训练集裁剪图做离线增强，提升样本多样性。",
    reviewQuestion: "确认增强是否执行且数量符合预期。",
    reviewAction: "重点看 generated_images 与 augment 输出目录。",
    expected: "完成后进入发布数据集。",
    checklist: [
      "增强图像数量是否合理。",
      "增强输出目录是否正确。",
    ],
  },
  publish_transfer: {
    label: "发布训练数据",
    goal: "生成训练 YAML，并把最终训练数据放到正确的位置。",
    reviewQuestion: "确认即将用于训练的数据版本、YAML 路径和发布位置是否正确。",
    reviewAction: "只看训练 YAML、数据版本、发布目录和推荐训练输出目录。",
    expected: "完成后会得到可直接训练的 YAML 和最终发布目录。",
    checklist: [
      "训练 YAML 路径是否正确。",
      "数据版本号是否符合本次 run。",
      "发布目录和后续训练目录是否正确。",
    ],
  },
  train: {
    label: "训练参数审核修改并确认",
    goal: "在真正开始训练前，做最后一次参数确认。",
    reviewQuestion: "确认现在能不能开始训练，以及训练参数是否合适。",
    reviewAction: "重点看底模、epochs、imgsz、训练 YAML 和输出目录。",
    expected: "确认后会提交异步训练任务，系统随后自动等待训练结束。",
    checklist: [
      "底模是否正确。",
      "训练轮数和图像尺寸是否符合目标。",
      "训练 YAML、输出目录、run 名称是否都正确。",
    ],
  },
  poll_train: {
    label: "训练",
    goal: "持续轮询训练任务，直到训练结束或失败。",
    reviewQuestion: "这里只需要知道训练是否顺利结束。",
    reviewAction: "优先看训练状态、退出码和输出目录。",
    expected: "训练成功后会进入最终验收；失败时会停在错误结果，供你排查。",
    checklist: [
      "训练状态是否成功。",
      "若失败，优先看退出码和日志。",
    ],
  },
  review_result: {
    label: "训练结果验收",
    goal: "根据训练结果决定这次 run 是否达到预期。",
    reviewQuestion: "确认这次训练结果是否可以接受并作为最终输出。",
    reviewAction: "优先看训练是否成功、输出目录是否生成、日志里有没有明显异常。",
    expected: "确认通过后会继续导出模型；如果不满意，就中止并调整参数后重新跑。",
    checklist: [
      "训练是否成功结束。",
      "输出目录是否生成在预期位置。",
      "日志中是否有明显异常、空结果或提前退出。",
    ],
  },
  export_model: {
    label: "模型导出",
    goal: "把训练得到的 best.pt 导出为 torchscript，供后续推理使用。",
    reviewQuestion: "这里只需要确认导出有没有成功完成。",
    reviewAction: "重点看导出状态、导出文件路径和退出码。",
    expected: "成功后会得到可用于后续推理的 torchscript 模型。",
    checklist: [
      "导出状态是否成功。",
      "导出模型路径是否存在。",
      "如果失败，优先看导出错误信息。",
    ],
  },
  model_infer: {
    label: "模型推理",
    goal: "使用导出的 torchscript 对单图/目录/多路径进行批量推理。",
    reviewQuestion: "确认模型路径、输入来源和阈值参数是否正确。",
    reviewAction: "重点看 model_path、source_path/source_paths、imgsz/conf/iou。",
    expected: "确认后会在 runs/infer 下生成标准推理产物目录。",
    checklist: [
      "模型路径是否为刚导出的 torchscript。",
      "输入路径是否覆盖本次待测数据。",
      "阈值与类别过滤是否符合当前场景。",
    ],
  },
};

const BUSINESS_STEP_TO_TECH_STEP = {
  label_transform_review: "review_labels",
};

const BASE_DISPLAY_STEPS = [
  "label_transform_review",
  "split_dataset",
  "publish_transfer",
  "train",
  "poll_train",
  "review_result",
  "export_model",
  "model_infer",
];

const DISPLAY_STEPS_BY_TECH_STEP = Object.entries(BUSINESS_STEP_TO_TECH_STEP).reduce((acc, [displayStep, techStep]) => {
  if (!acc[techStep]) acc[techStep] = [];
  acc[techStep].push(displayStep);
  return acc;
}, {});

function resolveActiveDisplayStep(activeStep) {
  if (!activeStep) return null;
  if (BASE_DISPLAY_STEPS.includes(activeStep) || activeStep === "crop_window" || activeStep === "augment_only") return activeStep;
  const displaySteps = DISPLAY_STEPS_BY_TECH_STEP[activeStep] || [];
  if (!displaySteps.length) return activeStep;
  return displaySteps[0];
}

function shouldShowSlidingWindowSteps(data) {
  if (data?.initial_params?.enable_sliding_window) return true;
  const stepResults = data?.step_results || {};
  return Boolean(stepResults.crop_window || stepResults.augment_only);
}

function shouldShowExportStep(data) {
  if (data?.initial_params?.yolo_export_after_train) return true;
  const stepResults = data?.step_results || {};
  return Boolean(stepResults.export_model);
}

function getDisplaySteps(data) {
  const steps = [...BASE_DISPLAY_STEPS];
  if (shouldShowSlidingWindowSteps(data)) {
    steps.splice(2, 0, "crop_window", "augment_only");
  }
  if (!shouldShowExportStep(data)) {
    return steps.filter((step) => step !== "export_model");
  }
  return steps;
}

function resolveStepResultForDisplay(stepName, data) {
  const stepResults = data.step_results || {};
  if (stepName === "label_transform_review") {
    const review = stepResults.review_labels;
    if (review) return review;
    if (stepResults.xml_to_yolo || stepResults.discover_classes) {
      return {
        status: "running",
        summary: "正在进行标注检查与转换修改。",
        data: {},
      };
    }
    return null;
  }
  return stepResults[stepName] || null;
}

const FIELD_LABELS = {
  class_names: "类别列表",
  class_counts: "类别统计",
  total_classes: "类别数",
  total_xml_files: "XML 文件数",
  converted_files: "已转换文件",
  total_boxes: "标注框总数",
  labels_dir: "标签目录",
  classes: "训练类别",
  class_to_id: "类别索引",
  classes_file: "classes.txt",
  output_dir: "输出目录",
  input_dir: "输入目录",
  split_output_dir: "划分目录",
  publish_input_dir: "发布源目录",
  crop_root: "裁剪输出目录",
  output_yaml_path: "训练 YAML",
  dataset_version: "数据版本",
  published_dataset_dir: "发布目录",
  recommended_train_project: "建议训练输出目录",
  recommended_train_name: "建议 Run 名称",
  model: "底模",
  epochs: "训练轮数",
  imgsz: "图像尺寸",
  yolo_train_env: "训练环境",
  yaml_path: "训练 YAML",
  project: "训练输出目录",
  name: "Run 名称",
  exit_code: "退出码",
  stdout: "标准输出",
  stderr: "错误输出",
  train_summary: "训练结论",
  train_task_id: "训练任务 ID",
  generated_crops: "生成窗口数",
  generated_labels: "生成标签数",
  generated_images: "增强新增图像",
  processed_images: "处理图像数",
  skipped_images: "跳过图像数",
  train_images: "训练图像",
  val_images: "验证图像",
  test_images: "测试图像",
  total_images: "总图像数",
  paired_images: "成对图像数",
  execution_mode: "执行模式",
  project_root_dir: "工作区目录",
  model_path: "模型路径",
  source_path: "输入路径",
  source_paths: "输入路径列表",
  run_args_path: "推理参数文件",
  summary_path: "推理汇总文件",
  detected_images: "命中图像数",
  no_detect_images: "未命中图像数",
  result_images: "结果图像数",
  labels_written: "标签文件数",
};

const STATUS_TEXT = {
  ok: "已完成",
  completed: "已完成",
  succeeded: "已完成",
  failed: "失败",
  error: "失败",
  skipped: "已跳过",
  running: "进行中",
  waiting: "待确认",
  interrupted: "待确认",
  pending: "未开始",
  unknown: "未知",
};

const MODEL_STEP_TOKENS = {
  training: ["train", "训练"],
  inference: ["predict", "infer", "inference", "推理"],
};

const qs = (id) => document.getElementById(id);

const el = {
  sopSelect: qs("sopSelect"),
  sopReviewProfile: qs("sopReviewProfile"),
  sopDescription: qs("sopDescription"),
  originalDataset: qs("originalDataset"),
  detectorName: qs("detectorName"),
  projectOwner: qs("projectOwner"),
  projectRootDir: qs("projectRootDir"),
  executionMode: qs("executionMode"),
  executionModeHint: qs("executionModeHint"),
  trainEnv: qs("trainEnv"),
  trainModel: qs("trainModel"),
  trainEpochs: qs("trainEpochs"),
  trainImgsz: qs("trainImgsz"),
  fullAccess: qs("fullAccess"),
  classMappingLaunch: qs("classMappingLaunch"),
  classMapFieldset: qs("classMapFieldset"),
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
  settingLocalYoloTrainEnv: qs("settingLocalYoloTrainEnv"),
  settingRemoteYoloTrainEnv: qs("settingRemoteYoloTrainEnv"),
  settingsEnvStatus: qs("settingsEnvStatus"),

  launchView: qs("launchView"),
  detailView: qs("detailView"),
  headerTitle: qs("headerTitle"),
  headerActions: qs("headerActions"),
  
  detailRunId: qs("detailRunId"),
  detailStatusPill: qs("detailStatusPill"),
  detailRefreshBtn: qs("detailRefreshBtn"),
  detailCopyBtn: qs("detailCopyBtn"),
  detailAbortBtn: qs("detailAbortBtn"),
  statusTimeline: qs("statusTimeline"),
  stepList: qs("stepList"),
  reviewCard: qs("reviewCard"),
  bottomActionArea: qs("bottomActionArea"),
  noReviewNote: qs("noReviewNote"),
  reviewStep: qs("reviewStep"),
  reviewHint: qs("reviewHint"),
  reviewQuestion: qs("reviewQuestion"),
  reviewAction: qs("reviewAction"),
  reviewExpectation: qs("reviewExpectation"),
  reviewChecklist: qs("reviewChecklist"),
  reviewAdvancedOverride: qs("reviewAdvancedOverride"),
  reviewAdvancedData: qs("reviewAdvancedData"),
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
  runningStepHintByRun: {},
  latestSnapshotByRun: {},
  timelineByRun: {},
  uiModeByRun: {},
  pollTimer: null,
  eventSource: null,
  activeTab: "local", // 'local' or 'remote'
  settings: loadJson("pipeline_settings", {
    localPath: "",
    remotePath: "",
    localYoloTrainEnv: "",
    remoteYoloTrainEnv: "",
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
    onSopChange();
  } catch (exc) {
    setStatus(el.statusBox, `加载 SOP 失败：${exc.message}`, "warn");
  }
}

function onSopChange() {
  const sopId = el.sopSelect.value;
  if (!sopId) {
    el.sopDescription.textContent = "选择上方 SOP 查看详情。";
    if (el.sopReviewProfile) {
      el.sopReviewProfile.innerHTML = "<option value=\"\">（使用 SOP 默认）</option>";
    }
    el.classMapFieldset?.classList.remove("sop-large-sliding");
    updateExecutionModeSettingHint();
    return;
  }
  const sop = sopCache.find((s) => s.id === sopId);
  if (!sop) return;
  el.sopDescription.textContent = sop.description;
  if (el.sopReviewProfile) {
    const profiles = sop.review_profiles || {};
    el.sopReviewProfile.innerHTML = "<option value=\"\">（使用 SOP 默认）</option>";
    for (const key of Object.keys(profiles)) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = key;
      el.sopReviewProfile.appendChild(opt);
    }
    if (sop.review_profile_default && profiles[sop.review_profile_default]) {
      el.sopReviewProfile.value = sop.review_profile_default;
    }
  }
  const d = sop.defaults || {};
  if (d.execution_mode) el.executionMode.value = d.execution_mode;
  if (d.yolo_train_model) el.trainModel.value = d.yolo_train_model;
  if (d.yolo_train_epochs) el.trainEpochs.value = d.yolo_train_epochs;
  if (d.yolo_train_imgsz) el.trainImgsz.value = d.yolo_train_imgsz;
  if (typeof d.full_access === "boolean") el.fullAccess.checked = d.full_access;
  if (el.classMapFieldset) {
    el.classMapFieldset.classList.toggle("sop-large-sliding", sopId === "local-large-sliding-window");
  }
  updateProjectRootDirPreview();
}


function setStatus(target, text, kind = "info") {
  target.textContent = text;
  target.className = `status ${kind}`;
}

function persistRuns() {
  saveJson(STORAGE_KEY, state.runs);
}

function normalizePathJoin(base, child) {
  const b = String(base || "").trim().replace(/\/+$|\\+$/g, "");
  const c = String(child || "").trim().replace(/^\/+|^\\+/, "");
  if (!b) return c;
  if (!c) return b;
  return `${b}/${c}`;
}

function getWorkspaceBasePathByMode(mode) {
  return mode === "local" ? (state.settings.localPath || "") : (state.settings.remotePath || "");
}

function modeNeedsWorkspacePath(mode) {
  const basePath = getWorkspaceBasePathByMode(mode);
  return !String(basePath || "").trim();
}

function updateExecutionModeSettingHint() {
  const mode = el.executionMode.value;
  const needsSetting = modeNeedsWorkspacePath(mode);
  el.executionMode.classList.toggle("needs-setting-pulse", needsSetting);
  if (needsSetting) {
    const target = mode === "local" ? "本地训练工作区路径" : "远程训练工作区路径";
    el.executionMode.title = `当前模式缺少配置：${target}。请先到“设置”中填写。`;
    el.executionModeHint.textContent = `当前 ${mode} 缺少 ${target}，请先到左下角“设置”中填写。`;
    el.executionModeHint.hidden = false;
  } else {
    el.executionMode.title = "";
    el.executionModeHint.hidden = true;
  }
}

function updateProjectRootDirPreview() {
  const owner = el.projectOwner.value.trim();
  const basePath = getWorkspaceBasePathByMode(el.executionMode.value);
  el.projectRootDir.value = normalizePathJoin(basePath, owner);
  updateExecutionModeSettingHint();
  refreshTrainEnvByMode();
}

function getYoloTrainEnvByMode(mode) {
  if (mode === "local") {
    return String(state.settings.localYoloTrainEnv || "").trim();
  }
  return String(state.settings.remoteYoloTrainEnv || "").trim();
}

function refreshTrainEnvByMode() {
  const mode = el.executionMode.value;
  const envValue = getYoloTrainEnvByMode(mode);
  if (envValue) el.trainEnv.value = envValue;
}

function buildRunPayload() {
  const owner = el.projectOwner.value.trim();
  const basePath = getWorkspaceBasePathByMode(el.executionMode.value);
  const projectRootDir = normalizePathJoin(basePath, owner);
  el.projectRootDir.value = projectRootDir;

  const yoloTrainEnv = getYoloTrainEnvByMode(el.executionMode.value) || el.trainEnv.value.trim();
  const payload = {
    original_dataset: el.originalDataset.value.trim(),
    detector_name: el.detectorName.value.trim(),
    project_root_dir: projectRootDir,
    execution_mode: el.executionMode.value,
    yolo_train_env: yoloTrainEnv,
    yolo_train_model: el.trainModel.value.trim(),
    yolo_train_epochs: Number(el.trainEpochs.value) || 100,
    yolo_train_imgsz: Number(el.trainImgsz.value) || 640,
    full_access: el.fullAccess.checked,
    self_api_url: location.origin,
  };

  if (el.classMappingLaunch) {
    Object.assign(payload, collectClassMappingPayload(el.classMappingLaunch));
  }
  if (el.sopSelect.value && el.sopReviewProfile?.value) {
    payload.review_profile = el.sopReviewProfile.value;
  }
  return payload;
}

function validatePayload(payload) {
  const required = ["original_dataset", "detector_name", "project_root_dir", "yolo_train_env"];
  if (!el.projectOwner.value.trim()) {
    throw new Error("字段 项目归属 必填");
  }
  if (modeNeedsWorkspacePath(el.executionMode.value)) {
    throw new Error("当前 execution_mode 对应工作区路径未设置，请先到设置里填写");
  }
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
    const err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    err.responseBody = body;
    err.httpStatus = resp.status;
    throw err;
  }
  return body;
}

async function waitForNextInterrupt(runId, initialData, options = {}) {
  const maxAttempts = options.maxAttempts || 8;
  const intervalMs = options.intervalMs || 350;
  let latest = initialData;
  for (let i = 0; i < maxAttempts; i++) {
    if (latest?.completed || latest?.interrupted) return latest;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
    try {
      latest = await fetchJson(`${API_BASE}/${runId}`);
    } catch {
      break;
    }
  }
  return latest;
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
      last_status: getDisplayStatusInfo(resp),
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
  if (resp.interrupted && resp.pending_review) {
    return "waiting";
  }
  return "running";
}

function getSnapshotOrderValue(snapshot = {}) {
  return typeof snapshot.revision === "number" ? snapshot.revision : -1;
}

function shouldApplySnapshot(snapshot, options = {}) {
  const runId = snapshot?.run_id;
  if (!runId) return false;
  if (options.force) return true;
  const prev = state.latestSnapshotByRun[runId];
  if (!prev) return true;
  const currentRevision = getSnapshotOrderValue(snapshot);
  const prevRevision = getSnapshotOrderValue(prev);
  if (currentRevision > prevRevision) return true;
  if (currentRevision < prevRevision) return false;
  if (snapshot.snapshot_id && prev.snapshot_id && snapshot.snapshot_id === prev.snapshot_id) {
    return false;
  }
  return true;
}

function applySnapshot(snapshot, options = {}) {
  if (!shouldApplySnapshot(snapshot, options)) return false;
  const runId = snapshot.run_id;
  state.latestSnapshotByRun[runId] = {
    revision: snapshot.revision ?? -1,
    snapshot_id: snapshot.snapshot_id || null,
  };
  if (snapshot.completed || snapshot.error) {
    delete state.uiModeByRun[runId];
  }
  appendTimelineEntry(snapshot);
  if (state.activeRunId === runId) {
    renderDetail(snapshot);
  } else {
    const runIdx = state.runs.findIndex((r) => r.run_id === runId);
    if (runIdx >= 0) {
      state.runs[runIdx].last_status = getDisplayStatusInfo(snapshot);
      persistRuns();
      renderRunList();
    }
  }
  return true;
}

function appendTimelineEntry(snapshot) {
  const runId = snapshot.run_id;
  if (!runId) return;
  const list = state.timelineByRun[runId] || [];
  const revision = typeof snapshot.revision === "number" ? snapshot.revision : -1;
  if (list.length && list[list.length - 1].revision === revision) return;
  const activeStep = snapshot.active_step || snapshot.current_step || "-";
  // 仅记录步骤变化：同一步骤内的状态刷新（等待/运行/轮询）不再重复追加。
  if (list.length && list[list.length - 1].active_step === activeStep) return;
  const statusKey = summarizeStatus(snapshot);
  list.push({
    revision,
    at: Date.now(),
    status: statusKey,
    active_step: activeStep,
    summary: snapshot.error || (snapshot.pending_review?.step ? `等待审核：${snapshot.pending_review.step}` : ""),
  });
  state.timelineByRun[runId] = list.slice(-120);
}

function renderStatusTimeline(runId) {
  if (!el.statusTimeline) return;
  const rows = state.timelineByRun[runId] || [];
  if (!rows.length) {
    el.statusTimeline.innerHTML = "<div class=\"status-timeline-item\"><span class=\"status-timeline-summary\">暂无状态变更记录</span></div>";
    return;
  }
  el.statusTimeline.innerHTML = "";
  for (const row of [...rows].reverse()) {
    const item = document.createElement("div");
    item.className = "status-timeline-item";
    const rev = document.createElement("span");
    rev.className = "status-timeline-revision";
    rev.textContent = `r${row.revision}`;
    const step = document.createElement("span");
    step.className = "status-timeline-step";
    step.textContent = row.active_step;
    const info = document.createElement("span");
    info.className = "status-timeline-summary";
    const timeText = new Date(row.at).toLocaleTimeString();
    info.textContent = `${getStatusText(row.status)} · ${timeText}${row.summary ? ` · ${row.summary}` : ""}`;
    item.append(rev, step, info);
    el.statusTimeline.appendChild(item);
  }
}

function prettyJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function getStepMeta(stepName) {
  return STEP_META[stepName] || {
    label: stepName,
    goal: "该步骤正在处理当前流程所需的数据。",
    reviewQuestion: "确认这一步的结果是否符合你的预期。",
    reviewAction: "优先看当前结论与关键输出路径。",
    expected: "确认后流程会继续推进到下一步。",
    checklist: ["确认当前结果是否满足后续步骤需要。"],
  };
}

function getStatusText(status) {
  return STATUS_TEXT[status] || status || STATUS_TEXT.unknown;
}

function detectModelOperation({ stepName = "", taskType = "", pendingStep = "" } = {}) {
  const haystack = [stepName, taskType, pendingStep]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  if (!haystack) return null;
  if (MODEL_STEP_TOKENS.inference.some((token) => haystack.includes(token))) {
    return "inference";
  }
  if (MODEL_STEP_TOKENS.training.some((token) => haystack.includes(token))) {
    return "training";
  }
  return null;
}

function getModelVerb(activity) {
  return activity === "inference" ? "推理" : "训练";
}

function buildModelTaskStatusInfo(runData, options = {}) {
  const currentStep = options.stepName || runData.current_step || "";
  const pendingStep = runData.pending_review?.step || "";
  const modelTask = runData.model_task;
  const activity = detectModelOperation({
    stepName: currentStep,
    taskType: modelTask?.task_type || "",
    pendingStep,
  });

  if (!activity) return null;

  const verb = getModelVerb(activity);
  if (runData.interrupted && runData.pending_review && currentStep === "review_result") {
    return {
      key: "waiting",
      text: `待确认${verb}结果`,
      summary: `${verb}已经完成，等待人工验收结果。`,
    };
  }

  if (
    runData.interrupted
    && runData.pending_review
    && (
      (activity === "training" && currentStep === "train")
      || (activity === "inference" && detectModelOperation({ stepName: currentStep }) === "inference")
    )
  ) {
    return {
      key: "waiting",
      text: `待确认${verb}参数`,
      summary: `${verb}参数等待人工确认。`,
    };
  }

  if (!modelTask) {
    if (currentStep === "poll_train") {
      return {
        key: "running",
        text: `等待${verb}结果`,
        summary: `${verb}任务已提交，系统正在等待完成。`,
      };
    }
    return null;
  }

  switch (modelTask.state) {
    case "pending":
      if ((modelTask.queue_position || 0) > 0) {
        const queueText = modelTask.queue_position > 1
          ? `，前面还有 ${modelTask.queue_position} 个任务`
          : "，即将自动开始";
        return {
          key: "queued",
          text: `已添加到${verb}队列`,
          summary: `当前任务已进入${verb}队列${queueText}。`,
        };
      }
      return {
        key: "pending",
        text: `等待${verb}启动`,
        summary: `${verb}任务已创建，正在等待启动。`,
      };
    case "running":
      return {
        key: "model-running",
        text: `模型${verb}中`,
        summary: `模型正在${verb}，当前任务已占用执行资源。`,
      };
    case "succeeded":
      return {
        key: "ok",
        text: `模型${verb}完成`,
        summary: `模型${verb}已经完成。`,
      };
    case "failed":
      return {
        key: "failed",
        text: `模型${verb}失败`,
        summary: `模型${verb}失败，需要查看日志排查。`,
      };
    case "cancelled":
      return {
        key: "skipped",
        text: `模型${verb}已取消`,
        summary: `模型${verb}任务已取消。`,
      };
    default:
      return null;
  }
}

function getDisplayStatusInfo(runData) {
  const uiMode = state.uiModeByRun[runData.run_id];
  if (uiMode === "confirming") {
    return {
      key: "running",
      text: "确认中",
      summary: "正在提交确认并等待最新状态",
    };
  }
  if (uiMode === "aborting") {
    return {
      key: "waiting",
      text: "中止中",
      summary: "正在提交中止请求",
    };
  }
  if (runData.completed && runData.error) {
    return {
      key: "failed",
      text: getStatusText("failed"),
      summary: "",
    };
  }
  const modelInfo = buildModelTaskStatusInfo(runData);
  if (modelInfo) return modelInfo;

  const statusKey = summarizeStatus(runData);
  return {
    key: statusKey,
    text: getStatusText(statusKey),
    summary: "",
  };
}

function buildModelStepResultStatusInfo(stepName, result, runData) {
  if (!result) return null;
  const activity = detectModelOperation({
    stepName,
    taskType: runData.model_task?.task_type || "",
    pendingStep: runData.pending_review?.step || "",
  });
  if (!activity) return null;

  const verb = getModelVerb(activity);
  if (stepName === "train" && result.status === "ok") {
    return { key: "ok", text: `已提交${verb}任务` };
  }
  if (stepName === "poll_train") {
    if (result.status === "ok") return { key: "ok", text: `模型${verb}完成` };
    if (result.status === "failed") return { key: "failed", text: `模型${verb}失败` };
    if (result.status === "skipped") return { key: "skipped", text: `模型${verb}已跳过` };
  }
  if (stepName === "review_result") {
    if (result.status === "ok") return { key: "ok", text: `${verb}结果已验收` };
    if (result.status === "failed") return { key: "failed", text: `${verb}结果待处理` };
  }
  return null;
}

function normalizeCachedStatus(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const key = value.key || value.status || "unknown";
    return {
      key,
      text: value.text || getStatusText(key),
    };
  }
  const key = typeof value === "string" ? value : "unknown";
  return {
    key,
    text: getStatusText(key),
  };
}

function labelForField(key) {
  return FIELD_LABELS[key] || key;
}

function formatFactValue(value, label = "") {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString("zh-CN") : String(value);
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "string") {
    const text = value.trim();
    if (!text) return "";
    if (text.length > 96 && !label.includes("日志")) return `${text.slice(0, 93)}...`;
    return text;
  }
  if (Array.isArray(value)) {
    if (!value.length) return "";
    const preview = value.slice(0, 4).map((item) => String(item)).join("、");
    return value.length > 4 ? `${preview} 等 ${value.length} 项` : preview;
  }
  if (typeof value === "object") {
    const keys = Object.keys(value);
    if (!keys.length) return "";
    if (label.includes("统计") || label.includes("索引")) {
      const preview = keys.slice(0, 3).map((key) => `${key}:${value[key]}`).join("，");
      return keys.length > 3 ? `${preview} 等 ${keys.length} 项` : preview;
    }
    return `共 ${keys.length} 项`;
  }
  return String(value);
}

function appendFact(facts, label, value) {
  const formatted = formatFactValue(value, label);
  if (!formatted) return;
  facts.push([label, formatted]);
}

function buildGenericFacts(source, limit = 6) {
  if (!source || typeof source !== "object") return [];
  const facts = [];
  for (const [key, value] of Object.entries(source)) {
    if (facts.length >= limit) break;
    if (["hint", "note", "instructions"].includes(key)) continue;
    appendFact(facts, labelForField(key), value);
  }
  return facts;
}

function buildStepFacts(stepName, result, source = {}, runData = {}) {
  const facts = [];
  const pollTrainData = (source?.train_data && typeof source.train_data === "object")
    ? source.train_data
    : source;
  const modelTask = runData.model_task;
  const modelInfo = buildModelTaskStatusInfo(runData, { stepName });

  switch (stepName) {
    case "label_transform_review":
      appendFact(facts, "类别数", runData.step_results?.discover_classes?.data?.total_classes);
      appendFact(facts, "已转换文件", runData.step_results?.xml_to_yolo?.data?.converted_files);
      appendFact(facts, "标签目录", runData.step_results?.xml_to_yolo?.data?.labels_dir);
      appendFact(facts, "最终训练类别", runData.step_results?.review_labels?.data?.classes);
      break;
    case "healthcheck":
      appendFact(facts, "当前结论", result?.status === "ok" ? "服务可用，可继续执行" : "服务异常，流程无法继续");
      appendFact(facts, "服务地址", runData.initial_params?.self_api_url || location.origin);
      break;
    case "discover_classes":
      appendFact(facts, "XML 文件数", source.total_xml_files);
      appendFact(facts, "发现类别", source.total_classes);
      appendFact(facts, "类别预览", source.class_names);
      break;
    case "xml_to_yolo":
      appendFact(facts, "已转换文件", source.total_xml_files ? `${source.converted_files}/${source.total_xml_files}` : source.converted_files);
      appendFact(facts, "标注框总数", source.total_boxes);
      appendFact(facts, "标签目录", source.labels_dir);
      appendFact(facts, "训练类别", source.classes);
      break;
    case "review_labels": {
      const classes = Array.isArray(source.classes) ? source.classes : Object.keys(source.class_to_id || {});
      appendFact(facts, "最终训练类别", classes);
      appendFact(facts, "类别数量", classes.length);
      appendFact(facts, "标签目录", source.labels_dir || runData.step_results?.xml_to_yolo?.data?.labels_dir);
      break;
    }
    case "split_dataset":
      appendFact(facts, "输出目录", source.output_dir);
      appendFact(facts, "训练图像", source.train_images);
      appendFact(facts, "验证图像", source.val_images);
      appendFact(facts, "测试图像", source.test_images);
      break;
    case "crop_window":
      appendFact(facts, "裁剪目录", source.crop_root || source.crop?.output_dir || source.output_dir);
      appendFact(facts, "生成窗口数", source.crop?.generated_crops || source.generated_crops);
      appendFact(facts, "生成标签数", source.crop?.generated_labels || source.generated_labels);
      break;
    case "augment_only":
      appendFact(facts, "增强目录", source.output_dir);
      appendFact(facts, "增强新增图像", source.generated_images);
      appendFact(facts, "处理图像数", source.processed_images);
      break;
    case "publish_transfer":
      appendFact(facts, "训练 YAML", source.output_yaml_path);
      appendFact(facts, "数据版本", source.dataset_version);
      appendFact(facts, "发布目录", source.published_dataset_dir);
      appendFact(facts, "建议训练输出目录", source.recommended_train_project);
      break;
    case "train":
      appendFact(facts, "底模", source.model);
      appendFact(facts, "训练轮数", source.epochs);
      appendFact(facts, "图像尺寸", source.imgsz);
      appendFact(facts, "训练 YAML", source.yaml_path);
      appendFact(facts, "训练输出目录", source.project);
      break;
    case "poll_train":
      appendFact(
        facts,
        "训练结论",
        result?.status === "ok"
          ? "训练已完成"
          : (modelInfo?.text || (result ? "训练失败" : "等待训练结果")),
      );
      appendFact(facts, "Run 名称", pollTrainData.name);
      appendFact(facts, "训练输出目录", pollTrainData.project);
      appendFact(facts, "退出码", pollTrainData.exit_code);
      appendFact(facts, "任务状态", modelTask?.state);
      appendFact(facts, "排队位置", modelTask?.queue_position);
      break;
    case "review_result":
      appendFact(facts, "训练结论", source.train_summary || result?.summary);
      appendFact(facts, "Run 名称", pollTrainData.name);
      appendFact(facts, "训练输出目录", pollTrainData.project);
      appendFact(facts, "退出码", pollTrainData.exit_code);
      break;
    case "export_model":
      appendFact(facts, "导出状态", source.export_status);
      appendFact(facts, "导出模型", source.export_file_path);
      appendFact(facts, "训练 YAML", source.dataset_yaml);
      appendFact(facts, "图像尺寸", source.imgsz);
      appendFact(facts, "退出码", source.exit_code);
      break;
    case "model_infer":
      appendFact(facts, "模型路径", source.model_path);
      appendFact(facts, "输出目录", source.output_dir);
      appendFact(facts, "总图像数", source.total_images);
      appendFact(facts, "命中图像数", source.detected_images);
      appendFact(facts, "未命中图像数", source.no_detect_images);
      break;
    default:
      return buildGenericFacts(source, 6);
  }

  return facts.filter(([, value]) => value !== "");
}

function buildStepSummary(stepName, result, data = {}) {
  const meta = getStepMeta(stepName);
  const modelInfo = state.currentRunData
    ? buildModelTaskStatusInfo(state.currentRunData, { stepName })
    : null;
  if (!result) return modelInfo?.summary || meta.goal;
  if (result.status === "failed") return result.summary || "这一步失败了，需要先处理问题再继续。";
  if (result.status === "skipped") return result.summary || "这一步已跳过。";

  switch (stepName) {
    case "label_transform_review":
      return result?.status === "ok"
        ? "标注检查、转换与类别确认已完成。"
        : "正在进行标注检查与转换修改。";
    case "healthcheck":
      return result.status === "ok" ? "服务可用，流程可以继续。" : "服务不可用，流程已停止。";
    case "discover_classes":
      return data.total_classes
        ? `已发现 ${data.total_classes} 个类别，下一步可确认是否需要合并或改名。`
        : "已完成类别扫描。";
    case "xml_to_yolo":
      return data.converted_files
        ? `标签已生成，可继续检查类别与索引是否正确。`
        : (result.summary || meta.goal);
    case "review_labels":
      return result.summary || "类别与标签口径已经确认。";
    case "split_dataset":
      return data.output_dir
        ? "训练集和验证集已经生成。"
        : (result.summary || meta.goal);
    case "crop_window":
      if (result?.status === "skipped") return result.summary || "滑窗裁剪已跳过。";
      return result?.status === "ok"
        ? "滑窗裁剪已完成。"
        : (result?.summary || "正在进行滑窗裁剪。");
    case "augment_only":
      if (result?.status === "skipped") return result.summary || "增强已跳过。";
      return result?.status === "ok"
        ? "增强已完成。"
        : (result?.summary || "正在进行增强。");
    case "publish_transfer":
      return data.output_yaml_path
        ? "训练 YAML 和发布目录已经就绪。"
        : (result.summary || meta.goal);
    case "train":
      if (modelInfo && state.currentRunData?.current_step === "train") {
        return modelInfo.summary || result.summary || "训练任务正在等待处理。";
      }
      return result.summary || "训练任务已提交，系统正在等待训练完成。";
    case "poll_train":
      if (modelInfo && state.currentRunData?.current_step === "poll_train") {
        return modelInfo.summary || "训练任务正在处理中。";
      }
      return result.status === "ok"
        ? "训练已经完成，可以查看最终验收结果。"
        : (result.summary || "训练失败，需要排查。");
    case "review_result":
      return result.summary || "训练结果已经验收。";
    case "export_model":
      if (result?.status === "skipped") return result.summary || "模型导出已跳过。";
      return result.summary || "模型导出已完成。";
    case "model_infer":
      return result.summary || "模型推理已完成。";
    default:
      return result.summary || meta.goal;
  }
}

function buildReviewModel(stepName, pending = {}, runData = {}) {
  const meta = getStepMeta(stepName);
  const review = pending.review && typeof pending.review === "object" ? pending.review : {};
  const checklist = Array.isArray(meta.checklist) ? meta.checklist : [];
  const defaultInstruction = "确认后点击 confirm（可在 params_override 中修改参数），中止请发 abort。";
  const hintFromPending = pending.instructions === defaultInstruction ? "" : (pending.instructions || "");
  return {
    title: meta.label,
    hint: hintFromPending || review.hint || "确认无误后继续；如果结果不符合预期，可修改参数或中止本次 run。",
    question: meta.reviewQuestion,
    action: meta.reviewAction,
    expectation: meta.expected,
    checklist,
    facts: buildStepFacts(stepName, null, review, runData),
  };
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

function renderFacts(container, facts, options = {}) {
  const emptyText = options.emptyText || "这里没有额外关键字段。通常只需要看上面的结论即可。";
  container.innerHTML = "";
  if (!facts.length) {
    const note = document.createElement("div");
    note.className = "agent-note";
    note.textContent = emptyText;
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

function createSelectField(label, key, value = "", choices = []) {
  const wrap = document.createElement("div");
  wrap.className = "review-form-field";
  const lb = document.createElement("label");
  lb.textContent = label;
  const select = document.createElement("select");
  select.dataset.overrideKey = key;
  const normalized = String(value || "").trim();
  for (const item of choices) {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    if (item.value === normalized) option.selected = true;
    select.appendChild(option);
  }
  wrap.append(lb, select);
  return wrap;
}

const REVIEW_FORM_SCHEMAS = {
  split_dataset: [
    { type: "text", label: "input_dir", key: "input_dir" },
    { type: "text", label: "output_dir", key: "output_dir" },
    {
      type: "select",
      label: "mode",
      key: "mode",
      choices: [
        { value: "train_val", label: "train_val" },
        { value: "train_val_test", label: "train_val_test" },
        { value: "train_only", label: "train_only" },
      ],
    },
    { type: "number-text", label: "train_ratio", key: "train_ratio", defaultValue: "0.85" },
    { type: "number-text", label: "val_ratio", key: "val_ratio", defaultValue: "0.15" },
    {
      type: "select",
      label: "shuffle",
      key: "shuffle",
      choices: [
        { value: "true", label: "true" },
        { value: "false", label: "false" },
      ],
      defaultValue: "true",
    },
    { type: "number", label: "seed", key: "seed", defaultValue: "42" },
    {
      type: "select",
      label: "copy_files",
      key: "copy_files",
      choices: [
        { value: "true", label: "true" },
        { value: "false", label: "false" },
      ],
      defaultValue: "true",
    },
  ],
  crop_window: [
    {
      type: "text",
      label: "input_dir",
      key: "input_dir",
      placeholder: "划分后数据集目录（通常是 split_output_dir）",
    },
    {
      type: "text",
      label: "output_dir",
      key: "output_dir",
      placeholder: "滑窗输出根目录（通常是 split_output_dir/crop）",
    },
    {
      type: "select",
      label: "only_wide",
      key: "only_wide",
      choices: [
        { value: "true", label: "true（仅裁剪宽图）" },
        { value: "false", label: "false（所有图都参与裁剪）" },
      ],
      defaultValue: "true",
    },
  ],
  augment_only: [
    {
      type: "text",
      label: "input_dir",
      key: "input_dir",
      placeholder: "增强输入目录（通常是 crop/train）",
    },
    {
      type: "text",
      label: "output_dir",
      key: "output_dir",
      placeholder: "增强输出目录（通常是 crop/train/augment）",
    },
  ],
};

function appendSchemaFields(grid, schema = [], review = {}) {
  for (const field of schema) {
    const currentValue = review[field.key];
    const finalValue = currentValue !== undefined && currentValue !== null
      ? String(currentValue)
      : (field.defaultValue || "");
    if (field.type === "select") {
      grid.append(
        createSelectField(field.label, field.key, finalValue, field.choices || []),
      );
      continue;
    }
    if (field.type === "number") {
      grid.append(
        createFormField(field.label, field.key, finalValue, {
          type: "number",
          placeholder: field.placeholder || "",
        }),
      );
      continue;
    }
    grid.append(
      createFormField(field.label, field.key, finalValue, {
        placeholder: field.placeholder || "",
      }),
    );
  }
}

function createClassesCheckboxField(classToId = {}, selectedClasses = []) {
  const wrap = document.createElement("div");
  wrap.className = "review-form-field review-form-field--full";
  const lb = document.createElement("label");
  lb.textContent = "classes（勾选类别）";
  wrap.appendChild(lb);

  const rows = Object.entries(classToId)
    .map(([name, idx]) => [Number(idx), name])
    .filter(([idx]) => Number.isFinite(idx))
    .sort((a, b) => a[0] - b[0]);
  const selected = new Set((selectedClasses || []).map((x) => Number(x)));

  const box = document.createElement("div");
  box.style.display = "grid";
  box.style.gap = "6px";
  box.style.marginTop = "6px";

  for (const [idx, name] of rows) {
    const line = document.createElement("label");
    line.style.display = "flex";
    line.style.alignItems = "center";
    line.style.gap = "8px";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.dataset.classesCheckbox = "1";
    cb.value = String(idx);
    cb.checked = selected.has(idx);
    const text = document.createElement("span");
    text.textContent = `[${idx}] ${name}`;
    line.append(cb, text);
    box.appendChild(line);
  }

  if (!rows.length) {
    const note = document.createElement("div");
    note.className = "agent-note";
    note.textContent = "未发现类别映射，无法生成勾选项。可用下方 classes(JSON数组) 手填。";
    box.appendChild(note);
  }

  wrap.appendChild(box);
  return wrap;
}

function renderOverrideForm(stepName, review = {}, initialParams = {}) {
  el.reviewOverrideForm.innerHTML = "";
  const formRoot = document.createElement("div");
  const title = document.createElement("h4");
  title.textContent = "需要调整时，在这里改参数";
  formRoot.appendChild(title);
  const grid = document.createElement("div");
  grid.className = "review-form-grid";

  if (stepName === "discover_classes") {
    const cmHost = document.createElement("div");
    cmHost.className = "review-class-mapping-host";
    grid.appendChild(cmHost);
    mountClassMappingForm(cmHost, {
      radioName: `reviewIdx_${Date.now()}`,
      discoveredNames: review.class_names || [],
      seed: {
        class_name_map: initialParams.class_name_map,
        final_classes: initialParams.final_classes,
        class_index_map: initialParams.class_index_map,
        training_names: initialParams.training_names,
      },
    });
  } else if (stepName === "review_labels") {
    const cmHost = document.createElement("div");
    cmHost.className = "review-class-mapping-host";
    grid.appendChild(cmHost);
    const reviewClassToId = review.class_to_id && typeof review.class_to_id === "object"
      ? review.class_to_id
      : undefined;
    const reviewClasses = Array.isArray(review.classes) ? review.classes : undefined;
    const indexSeed = reviewClassToId && Object.keys(reviewClassToId).length
      ? reviewClassToId
      : initialParams.class_index_map;
    const trainingSeed = reviewClasses && reviewClasses.length
      ? reviewClasses
      : initialParams.training_names;
    mountClassMappingForm(cmHost, {
      radioName: `reviewLabelsIdx_${Date.now()}`,
      discoveredNames: reviewClassToId ? Object.keys(reviewClassToId) : [],
      seed: {
        class_name_map: review.class_name_map ?? initialParams.class_name_map,
        final_classes: review.final_classes ?? initialParams.final_classes,
        class_index_map: review.class_index_map ?? indexSeed,
        training_names: review.training_names ?? trainingSeed,
      },
    });
  } else if (stepName === "publish_transfer") {
    grid.append(
      createFormField("publish_input_dir", "input_dir", review.publish_input_dir || ""),
      createFormField("execution_mode", "publish_mode", review.execution_mode || "local"),
      createFormField("project_root_dir", "project_root_dir", review.project_root_dir || ""),
      createFormField("detector_name", "detector_name", review.detector_name || ""),
    );
  } else if (REVIEW_FORM_SCHEMAS[stepName]) {
    appendSchemaFields(grid, REVIEW_FORM_SCHEMAS[stepName], review);
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
  } else if (stepName === "model_infer") {
    const classToId =
      (initialParams.class_to_id && typeof initialParams.class_to_id === "object" && initialParams.class_to_id)
      || (review.class_to_id && typeof review.class_to_id === "object" && review.class_to_id)
      || (state.currentRunData?.step_results?.review_labels?.data?.class_to_id)
      || (state.currentRunData?.step_results?.xml_to_yolo?.data?.class_to_id)
      || {};
    grid.append(
      createFormField("yolo_train_env", "yolo_train_env", review.yolo_train_env || ""),
      createFormField("model_path", "model_path", review.model_path || "", { full: true }),
      createFormField("source_path", "source_path", review.source_path || "", { full: true }),
      createFormField(
        "source_paths(多行，一行一个路径)",
        "source_paths",
        Array.isArray(review.source_paths) ? review.source_paths.join("\n") : "",
        { full: true, multiline: true, rows: 2 },
      ),
      createFormField("project", "project", review.project || "", { full: true }),
      createFormField("name", "name", review.name || ""),
      createFormField("imgsz", "imgsz", String(review.imgsz ?? 640), { type: "number" }),
      createFormField("conf", "conf", String(review.conf ?? 0.25)),
      createFormField("iou", "iou", String(review.iou ?? 0.7)),
    );
    grid.append(createClassesCheckboxField(classToId, review.classes || []));
    grid.append(
      createFormField(
        "classes(逗号分隔，兜底)",
        "classes",
        Array.isArray(review.classes) ? review.classes.join(",") : "",
      ),
    );
  } else {
    grid.append(createFormField("备注（可选）", "_note", "", {
      multiline: true,
      rows: 2,
      full: true,
      placeholder: "这个审核点通常不需要改单独参数；如果确实需要，再展开下方高级参数编辑。",
    }));
  }

  formRoot.appendChild(grid);
  el.reviewOverrideForm.appendChild(formRoot);
  el.reviewOverrideForm.hidden = false;
}

function collectOverrideFormValues() {
  const override = {};
  const cmRoot = el.reviewOverrideForm.querySelector(".class-mapping-form");
  if (cmRoot) {
    Object.assign(override, collectClassMappingPayload(cmRoot));
  }
  const fields = el.reviewOverrideForm.querySelectorAll("[data-override-key]");
  fields.forEach((input) => {
    const key = input.dataset.overrideKey;
    const raw = String(input.value || "").trim();
    if (!raw || key === "_note") return;
    if (key === "epochs" || key === "imgsz") {
      override[key] = Number(raw);
      return;
    }
    if (key === "conf" || key === "iou") {
      override[key] = Number(raw);
      return;
    }
    if (key === "train_ratio" || key === "val_ratio") {
      override[key] = Number(raw);
      return;
    }
    if (key === "seed") {
      override[key] = Number(raw);
      return;
    }
    if (key === "shuffle" || key === "copy_files") {
      override[key] = raw.toLowerCase() === "true";
      return;
    }
    if (key === "only_wide") {
      override[key] = raw.toLowerCase() === "true";
      return;
    }
    if (key === "source_paths") {
      override[key] = raw
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
      return;
    }
    if (key === "classes") {
      override[key] = raw
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean)
        .map((x) => Number(x))
        .filter((x) => Number.isFinite(x));
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
  const classChecks = el.reviewOverrideForm.querySelectorAll("[data-classes-checkbox='1']");
  if (classChecks.length > 0) {
    const picked = Array.from(classChecks)
      .filter((n) => n.checked)
      .map((n) => Number(n.value))
      .filter((n) => Number.isFinite(n));
    override.classes = picked;
  }
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
    const statusInfo = normalizeCachedStatus(run.last_status);
    const pill = document.createElement("span");
    pill.className = `pill ${statusInfo.key || ""}`;
    pill.textContent = statusInfo.text;
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

  // 根据当前 tab 选择默认 execution_mode，再按项目归属拼接 project_root_dir
  if (state.activeTab === "local") {
    el.executionMode.value = "local";
  } else if (state.activeTab === "remote") {
    el.executionMode.value = "remote_sftp";
  }
  refreshTrainEnvByMode();
  updateProjectRootDirPreview();
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
  applySnapshot(data, { force: true });
  renderStatusTimeline(runId);
  if (!data.completed) {
    startPolling(runId);
  }
}

function renderDetail(data) {
  state.currentRunData = data;
  renderStatusTimeline(data.run_id);
  if (data.interrupted && data.pending_review?.step) {
    state.runningStepHintByRun[data.run_id] = data.pending_review.step;
  } else if (data.completed || data.error) {
    delete state.runningStepHintByRun[data.run_id];
  }
  const hintedStep = !data.interrupted ? state.runningStepHintByRun[data.run_id] : null;
  const statusInfo = getDisplayStatusInfo(data);
  el.detailStatusPill.textContent = statusInfo.text;
  el.detailStatusPill.className = `pill ${statusInfo.key}`;
  const rawActiveStep = data.active_step || hintedStep || data.current_step;
  const activeDisplayStep = resolveActiveDisplayStep(rawActiveStep);
  const displaySteps = getDisplaySteps(data);

  el.stepList.innerHTML = "";
  for (const stepName of displaySteps) {
    const result = resolveStepResultForDisplay(stepName, data);
    const li = document.createElement("li");
    let cls = "";
    let statusInfoForStep = { key: "pending", text: getStatusText("pending") };
    if (result) {
      if (result.status === "ok") { cls = "step--ok"; statusInfoForStep = { key: "ok", text: getStatusText("ok") }; }
      else if (result.status === "failed") { cls = "step--failed"; statusInfoForStep = { key: "failed", text: getStatusText("failed") }; }
      else if (result.status === "skipped") { cls = "step--skipped"; statusInfoForStep = { key: "skipped", text: getStatusText("skipped") }; }
      const modelResultStatus = buildModelStepResultStatusInfo(stepName, result, data);
      if (modelResultStatus) {
        statusInfoForStep = modelResultStatus;
      }
    }
    if (activeDisplayStep === stepName && !result) {
      cls = "step--running"; statusInfoForStep = { key: "running", text: getStatusText("running") };
    }
    if (data.interrupted && data.pending_review && activeDisplayStep === stepName) {
      cls = "step--waiting"; statusInfoForStep = { key: "waiting", text: getStatusText("waiting") };
    }

    const modelStatusForStep = buildModelTaskStatusInfo(data, { stepName });
    if (modelStatusForStep && activeDisplayStep === stepName) {
      statusInfoForStep = {
        key: modelStatusForStep.key,
        text: modelStatusForStep.text,
      };
    }
    li.className = cls;

    const details = document.createElement("details");
    details.className = "step-card";
    if (
      (data.interrupted && activeDisplayStep === stepName) ||
      (activeDisplayStep === stepName && !result) ||
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
    name.textContent = getStepMeta(stepName).label;
    name.title = stepName;
    const pill = document.createElement("span");
    pill.className = `pill ${statusInfoForStep.key}`;
    pill.textContent = statusInfoForStep.text;
    header.append(name, pill);

    const summaryText = document.createElement("div");
    summaryText.className = "step__summary";
    summaryText.textContent = buildStepSummary(stepName, result, result?.data || {});
    summaryLine.append(header, summaryText);

    const detailData = result?.data || {};
    const detailBody = document.createElement("div");
    detailBody.className = "step-detail";
    const meta = getStepMeta(stepName);
    const guide = document.createElement("div");
    guide.className = "step-guide";
    const guideGoal = document.createElement("p");
    guideGoal.className = "step-guide-text";
    guideGoal.textContent = `这一步的作用：${meta.goal}`;
    const guideResult = document.createElement("p");
    guideResult.className = "step-guide-text";
    guideResult.textContent = `当前结论：${buildStepSummary(stepName, result, detailData)}`;
    guide.append(guideGoal, guideResult);
    detailBody.appendChild(guide);
    const factsWrap = document.createElement("div");
    renderFacts(
      factsWrap,
      buildStepFacts(stepName, result, detailData, data),
      {
        emptyText: "这个步骤没有额外需要你阅读的字段。通常只看上面的当前结论即可。",
      },
    );
    detailBody.appendChild(factsWrap);

    const rawDetails = document.createElement("details");
    rawDetails.className = "disclosure";
    const rawSummary = document.createElement("summary");
    rawSummary.textContent = "排查信息（输入 / 输出 / 原始 JSON，仅异常时需要）";
    const { input, output } = extractStepIo(detailData);
    const rawPre = document.createElement("pre");
    rawPre.className = "step-detail-json";
    rawPre.textContent = prettyJson({
      status: result?.status || statusInfoForStep.key,
      model_task: data.model_task || null,
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
    const reviewModel = buildReviewModel(stepName, pending, data);
    el.reviewStep.textContent = reviewModel.title;
    el.reviewHint.textContent = reviewModel.hint;
    el.reviewHint.hidden = false;
    if (el.reviewQuestion) el.reviewQuestion.textContent = reviewModel.question;
    if (el.reviewAction) el.reviewAction.textContent = reviewModel.action;
    if (el.reviewExpectation) el.reviewExpectation.textContent = reviewModel.expectation;
    const dataBlock = { ...(pending.review || {}) };
    if (el.reviewFacts) {
      el.reviewFacts.className = "review-facts";
      renderFacts(el.reviewFacts, reviewModel.facts, {
        emptyText: "这个审核点没有额外关键字段。通常只需要按上面的判断说明操作。",
      });
    }
    if (el.reviewChecklist) {
      el.reviewChecklist.innerHTML = "";
      for (const item of reviewModel.checklist) {
        const li = document.createElement("li");
        li.textContent = item;
        el.reviewChecklist.appendChild(li);
      }
    }
    el.reviewData.textContent = JSON.stringify(dataBlock, null, 2);
    if (el.reviewAdvancedOverride) el.reviewAdvancedOverride.open = false;
    if (el.reviewAdvancedData) el.reviewAdvancedData.open = false;
    el.reviewCard.classList.remove("focus-mode");
    renderOverrideForm(stepName, dataBlock, data.initial_params || {});
  } else if (data.interrupted && !data.pending_review) {
    el.reviewCard.hidden = false;
    el.noReviewNote.hidden = true;
    el.bottomActionArea.hidden = true;
    el.reviewOverrideForm.hidden = true;
    el.reviewOverrideForm.innerHTML = "";
    if (el.reviewFacts) el.reviewFacts.innerHTML = "";
    if (el.reviewChecklist) el.reviewChecklist.innerHTML = "";
    el.reviewStep.textContent = "待确认步骤";
    if (el.reviewQuestion) el.reviewQuestion.textContent = "正在加载确认表单…";
    if (el.reviewAction) el.reviewAction.textContent = "请稍候，系统会自动刷新并显示可修改参数。";
    if (el.reviewExpectation) el.reviewExpectation.textContent = "确认表单就绪后，可直接修改参数并点击确认。";
    el.reviewHint.textContent = "当前处于待确认状态，但审核数据尚未返回，正在重试获取。";
    el.reviewHint.hidden = false;
    if (!state.pollTimer && !state.eventSource && state.activeRunId) {
      startPolling(state.activeRunId);
    }
  } else {
    el.reviewCard.hidden = true;
    el.bottomActionArea.hidden = true;
    el.reviewOverrideForm.hidden = true;
    el.reviewOverrideForm.innerHTML = "";
    if (el.reviewFacts) el.reviewFacts.innerHTML = "";
    if (el.reviewChecklist) el.reviewChecklist.innerHTML = "";
    if (el.reviewQuestion) el.reviewQuestion.textContent = "";
    if (el.reviewAction) el.reviewAction.textContent = "";
    if (el.reviewExpectation) el.reviewExpectation.textContent = "";
    el.reviewHint.hidden = false;
    el.reviewCard.classList.remove("focus-mode");
    el.noReviewNote.hidden = false;
    if (data.completed) {
      el.noReviewNote.textContent = data.error
        ? `流程已终止：${data.error}`
        : "流程已完成。建议重点查看最后的训练结果与输出目录。";
    } else {
      el.noReviewNote.textContent = "当前不需要你操作，流程会自动继续。";
    }
  }

  const runIdx = state.runs.findIndex((r) => r.run_id === data.run_id);
  if (runIdx >= 0) {
    state.runs[runIdx].last_status = statusInfo;
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
          applySnapshot(JSON.parse(ev.data));
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
      applySnapshot(data);
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
  stopPolling();
  state.uiModeByRun[state.activeRunId] = "confirming";
  renderRunList();
  const pendingStep = state.currentRunData?.pending_review?.step;
  if (pendingStep) {
    state.runningStepHintByRun[state.activeRunId] = pendingStep;
  }
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
    let data = await fetchJson(
      `${API_BASE}/${state.activeRunId}/confirm`,
      {
        method: "POST",
        body: JSON.stringify({ decision: "confirm", params_override: override }),
      },
    );
    applySnapshot(data, { force: true });
    if (!data.completed && !data.interrupted) {
      data = await waitForNextInterrupt(state.activeRunId, data);
    }
    applySnapshot(data, { force: true });
    el.reviewOverride.value = "";
    setStatus(el.reviewStatus, "已确认，流程继续。", "success");
    if (!data.completed) startPolling(state.activeRunId);
  } catch (exc) {
    setStatus(el.reviewStatus, `确认失败：${exc.message}`, "error");
  } finally {
    el.confirmBtn.disabled = false;
    delete state.uiModeByRun[state.activeRunId];
  }
}

async function abortCurrentStep() {
  if (!state.activeRunId) return;
  if (!confirm("确认要中止当前 run 吗？")) return;
  stopPolling();
  state.uiModeByRun[state.activeRunId] = "aborting";
  renderRunList();
  el.abortStepBtn.disabled = true;
  if (el.detailAbortBtn) el.detailAbortBtn.disabled = true;
  try {
    const data = await fetchJson(
      `${API_BASE}/${state.activeRunId}/abort`,
      { method: "POST" },
    );
    applySnapshot(data, { force: true });
    if (data.completed || data.error) {
      setStatus(el.reviewStatus, "run 已中止。", "warn");
      stopPolling();
    } else {
      setStatus(el.reviewStatus, "已发送中止请求，等待当前步骤收尾后停止。", "warn");
      startPolling(state.activeRunId);
    }
  } catch (exc) {
    setStatus(el.reviewStatus, `中止失败：${exc.message}`, "error");
  } finally {
    el.abortStepBtn.disabled = false;
    if (el.detailAbortBtn) el.detailAbortBtn.disabled = false;
    delete state.uiModeByRun[state.activeRunId];
  }
}

async function refreshAll() {
  for (const run of state.runs) {
    try {
      const data = await fetchJson(`${API_BASE}/${run.run_id}`);
      applySnapshot(data);
    } catch {
      run.last_status = { key: "unknown", text: getStatusText("unknown") };
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
  const mode = p.execution_mode || "local";
  state.activeTab = mode === "local" ? "local" : "remote";
  showLaunchView();

  if (p.original_dataset) el.originalDataset.value = p.original_dataset;
  if (p.detector_name) el.detectorName.value = p.detector_name;
  if (p.execution_mode) el.executionMode.value = p.execution_mode;
  if (p.project_root_dir) {
    const basePath = getWorkspaceBasePathByMode(el.executionMode.value);
    const fullPath = String(p.project_root_dir);
    if (basePath && fullPath.startsWith(basePath)) {
      el.projectOwner.value = fullPath.slice(basePath.length).replace(/^\/+/, "") || "";
    } else {
      el.projectOwner.value = fullPath.split("/").filter(Boolean).pop() || "";
    }
  }
  if (p.yolo_train_env) el.trainEnv.value = p.yolo_train_env;
  if (p.yolo_train_model) el.trainModel.value = p.yolo_train_model;
  if (p.yolo_train_epochs) el.trainEpochs.value = p.yolo_train_epochs;
  if (p.yolo_train_imgsz) el.trainImgsz.value = p.yolo_train_imgsz;
  if (typeof p.full_access === "boolean") el.fullAccess.checked = p.full_access;

  if (el.classMappingLaunch) {
    mountClassMappingForm(el.classMappingLaunch, {
      radioName: "launchIdxMode",
      seed: {
        class_name_map: p.class_name_map,
        final_classes: p.final_classes,
        class_index_map: p.class_index_map,
        training_names: p.training_names,
      },
    });
  }

  updateProjectRootDirPreview();
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
  if (el.settingLocalYoloTrainEnv) {
    el.settingLocalYoloTrainEnv.value = state.settings.localYoloTrainEnv || "";
  }
  if (el.settingRemoteYoloTrainEnv) {
    el.settingRemoteYoloTrainEnv.value = state.settings.remoteYoloTrainEnv || "";
  }
  if (el.settingsEnvStatus) {
    el.settingsEnvStatus.textContent = "保存时会自动校验本地/远程 yolo_train_env 的 ultralytics/torch/cv2 等依赖。";
  }
  el.settingsModal.hidden = false;
});
el.closeSettingsBtn.addEventListener("click", () => {
  el.settingsModal.hidden = true;
});
el.saveSettingsBtn.addEventListener("click", async () => {
  const nextLocalYoloEnv = String(el.settingLocalYoloTrainEnv?.value || "").trim();
  const nextRemoteYoloEnv = String(el.settingRemoteYoloTrainEnv?.value || "").trim();
  if (!nextLocalYoloEnv) {
    if (el.settingsEnvStatus) el.settingsEnvStatus.textContent = "请至少填写本地 yolo_train_env。";
    return;
  }
  if (el.settingsEnvStatus) el.settingsEnvStatus.textContent = "正在校验本地 yolo_train_env ...";
  let validatedLocal;
  try {
    validatedLocal = await fetchJson("/api/v1/validate-yolo-env", {
      method: "POST",
      body: JSON.stringify({ yolo_train_env: nextLocalYoloEnv }),
    });
  } catch (exc) {
    const backendDetail = exc?.responseBody?.detail;
    const extra = typeof backendDetail === "string" ? `\n${backendDetail}` : "";
    if (el.settingsEnvStatus) el.settingsEnvStatus.textContent = `本地 yolo_train_env 校验失败：${exc.message}${extra}`;
    return;
  }
  if (!validatedLocal || validatedLocal.status !== "ok") {
    if (el.settingsEnvStatus) {
      el.settingsEnvStatus.textContent = `本地 yolo_train_env 校验失败：\n${validatedLocal?.stderr || validatedLocal?.stdout || "依赖不可用"}`;
    }
    return;
  }
  if (nextRemoteYoloEnv) {
    if (el.settingsEnvStatus) el.settingsEnvStatus.textContent = "正在校验远程 yolo_train_env ...";
    let validatedRemote;
    try {
      validatedRemote = await fetchJson("/api/v1/validate-yolo-env", {
        method: "POST",
        body: JSON.stringify({ yolo_train_env: nextRemoteYoloEnv }),
      });
    } catch (exc) {
      const backendDetail = exc?.responseBody?.detail;
      const extra = typeof backendDetail === "string" ? `\n${backendDetail}` : "";
      if (el.settingsEnvStatus) el.settingsEnvStatus.textContent = `远程 yolo_train_env 校验失败：${exc.message}${extra}`;
      return;
    }
    if (!validatedRemote || validatedRemote.status !== "ok") {
      if (el.settingsEnvStatus) {
        el.settingsEnvStatus.textContent = `远程 yolo_train_env 校验失败：\n${validatedRemote?.stderr || validatedRemote?.stdout || "依赖不可用"}`;
      }
      return;
    }
  }

  state.settings.localPath = el.settingLocalPath.value.trim();
  state.settings.remotePath = el.settingRemotePath.value.trim();
  state.settings.localYoloTrainEnv = nextLocalYoloEnv;
  state.settings.remoteYoloTrainEnv = nextRemoteYoloEnv;
  saveJson("pipeline_settings", state.settings);
  if (el.settingsEnvStatus) {
    el.settingsEnvStatus.textContent = nextRemoteYoloEnv
      ? "本地/远程 yolo_train_env 校验通过，已保存。"
      : "本地 yolo_train_env 校验通过，已保存（远程未填写）。";
  }
  refreshTrainEnvByMode();
  el.settingsModal.hidden = true;
  showLaunchView(); // Update form if needed
  updateExecutionModeSettingHint();
});

el.projectOwner.addEventListener("input", updateProjectRootDirPreview);
el.executionMode.addEventListener("change", updateProjectRootDirPreview);

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
if (el.classMappingLaunch) {
  mountClassMappingForm(el.classMappingLaunch, { radioName: "launchIdxMode", seed: {} });
}
if (state.runs[0]) {
  openRun(state.runs[0].run_id);
} else {
  showLaunchView();
}
updateExecutionModeSettingHint();
