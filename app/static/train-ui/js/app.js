/**
 * 页面入口：串联状态、表单、预览与提交。
 */

import { DEFAULT_REMOTE_HOST } from "./config.js";
import {
  createTrainUiState,
  persistWorkspaces,
  persistProjects,
  persistRemoteSnapshot,
  persistRunTargetPref,
  persistSelfApiUrl,
} from "./state.js";
import { buildTrainingWorkflowPayload, hasFullSsh } from "./payload.js";
import {
  el,
  showStatus,
  renderWorkspaces,
  renderProjectMeta,
  readFormFieldsForPayload,
  getSelectedRunTarget,
  setRunTargetRadios,
  updateCredentialHint,
  updateRunSummaryFromPayload,
} from "./views.js";

export function initTrainUiPage() {
  const state = createTrainUiState();

  function ensureDefaultProjects() {
    if (!Object.keys(state.projects).length) {
      state.projects = { TVDS: ["nzxj_louyou"] };
    }
  }

  function syncPreview() {
    const runTarget = getSelectedRunTarget();
    persistRunTargetPref(runTarget);
    state.runTargetPref = runTarget;

    const fields = readFormFieldsForPayload(runTarget);
    const payload = buildTrainingWorkflowPayload(fields);
    const preview = el("payloadPreview");
    if (preview) {
      preview.value = JSON.stringify(payload, null, 2);
    }

    updateRunSummaryFromPayload(payload);

    const sshReady = hasFullSsh(fields);
    updateCredentialHint(sshReady);

    const modeLabel = el("executionMode");
    if (modeLabel) {
      modeLabel.textContent =
        runTarget === "remote_sftp" ? "remote_sftp（本次提交）" : "local（本次提交）";
    }

    const remoteRadio = document.querySelector('input[name="runTarget"][value="remote_sftp"]');
    if (remoteRadio) {
      remoteRadio.disabled = !sshReady;
      if (!sshReady && runTarget === "remote_sftp") {
        setRunTargetRadios("local");
        syncPreview();
      }
    }
  }

  const workspaceHandlers = {
    onUse(workspace) {
      el("workspaceRootDir").value = workspace;
      syncPreview();
    },
    onRemove(workspace) {
      state.workspaces = state.workspaces.filter((item) => item !== workspace);
      persistWorkspaces(state);
      renderWorkspaces(state, workspaceHandlers);
      syncPreview();
    },
  };

  function renderProjects() {
    ensureDefaultProjects();
    const projectSelect = el("projectName");
    const detectorSelect = el("detectorName");
    if (!projectSelect || !detectorSelect) return;

    const names = Object.keys(state.projects).sort();
    const currentProject =
      projectSelect.value && state.projects[projectSelect.value] ? projectSelect.value : names[0];

    projectSelect.innerHTML = names.map((name) => `<option value="${name}">${name}</option>`).join("");
    projectSelect.value = currentProject;

    const detectors = state.projects[currentProject] || [];
    detectorSelect.innerHTML = detectors.map((name) => `<option value="${name}">${name}</option>`).join("");
    if (!detectors.length) {
      detectorSelect.innerHTML = `<option value="">请先添加 detector</option>`;
    }
    renderProjectMeta(state, currentProject);
    syncPreview();
  }

  function applyStoredSelfApiUrlToInput() {
    const input = el("selfApiUrl");
    if (!input) return;
    const stored = (state.selfApiUrlStored || "").trim();
    input.value = stored || window.location.origin;
  }

  ensureDefaultProjects();
  setRunTargetRadios(state.runTargetPref || "local");

  applyStoredSelfApiUrlToInput();

  if (el("workspaceRootDir")) {
    el("workspaceRootDir").value = state.workspaces[0] || "";
  }

  const rh = el("remoteHost");
  if (rh) {
    const saved = (state.remote.remote_host || "").trim();
    rh.value = saved || DEFAULT_REMOTE_HOST;
  }
  if (el("remoteUsername")) {
    el("remoteUsername").value = state.remote.remote_username || "";
  }
  if (el("remoteKeyPath")) {
    el("remoteKeyPath").value = state.remote.remote_private_key_path || "";
  }
  if (el("remoteWorkspaceRootDir")) {
    el("remoteWorkspaceRootDir").value = state.remote.remote_project_root_dir || "";
  }

  renderWorkspaces(state, workspaceHandlers);
  renderProjects();

  const settingsDlg = el("settingsDialog");

  const openSettingsBtn = el("openSettingsBtn");
  openSettingsBtn?.addEventListener("click", () => {
    settingsDlg?.showModal();
    openSettingsBtn.setAttribute("aria-expanded", "true");
  });

  const closeSettings = () => {
    settingsDlg?.close();
    openSettingsBtn?.setAttribute("aria-expanded", "false");
  };

  el("closeSettingsBtn")?.addEventListener("click", () => {
    closeSettings();
  });

  el("closeSettingsX")?.addEventListener("click", () => {
    closeSettings();
  });

  settingsDlg?.addEventListener("close", () => {
    openSettingsBtn?.setAttribute("aria-expanded", "false");
  });

  settingsDlg?.addEventListener("click", (e) => {
    if (e.target === settingsDlg) closeSettings();
  });

  el("resetSelfApiUrlBtn")?.addEventListener("click", () => {
    const input = el("selfApiUrl");
    if (input) input.value = window.location.origin;
    persistSelfApiUrl("");
    state.selfApiUrlStored = "";
    syncPreview();
    showStatus("已恢复为当前页面 origin。", "info");
  });

  function persistSettingsFromForm() {
    const apiInput = el("selfApiUrl");
    const raw = apiInput?.value.trim() ?? "";
    const origin = window.location.origin;
    if (!raw || raw === origin) {
      persistSelfApiUrl("");
      state.selfApiUrlStored = "";
      if (apiInput) {
        apiInput.value = origin;
      }
    } else {
      persistSelfApiUrl(raw);
      state.selfApiUrlStored = raw;
    }

    persistRemoteSnapshot({
      remote_host: el("remoteHost")?.value.trim() ?? "",
      remote_username: el("remoteUsername")?.value.trim() ?? "",
      remote_private_key_path: el("remoteKeyPath")?.value.trim() ?? "",
      remote_project_root_dir: el("remoteWorkspaceRootDir")?.value.trim() ?? "",
    });
    state.remote = {
      remote_host: el("remoteHost")?.value.trim() ?? "",
      remote_username: el("remoteUsername")?.value.trim() ?? "",
      remote_private_key_path: el("remoteKeyPath")?.value.trim() ?? "",
      remote_project_root_dir: el("remoteWorkspaceRootDir")?.value.trim() ?? "",
    };
  }

  el("saveSettingsBtn")?.addEventListener("click", () => {
    persistSettingsFromForm();
    syncPreview();
    closeSettings();
    showStatus("设置已保存（浏览器本地）。", "info");
  });

  el("saveWorkspace")?.addEventListener("click", () => {
    const value = el("workspaceRootDir")?.value.trim();
    if (!value) return showStatus("workspace_root_dir 不能为空。", "warn");
    if (!state.workspaces.includes(value)) {
      state.workspaces.unshift(value);
    }
    state.workspaces = state.workspaces.slice(0, 8);
    persistWorkspaces(state);
    renderWorkspaces(state, workspaceHandlers);
    syncPreview();
    showStatus("工作区已加入历史。", "info");
  });

  el("clearWorkspaces")?.addEventListener("click", () => {
    state.workspaces = [];
    persistWorkspaces(state);
    renderWorkspaces(state, workspaceHandlers);
    syncPreview();
    showStatus("工作区历史已清空。", "info");
  });

  el("addProject")?.addEventListener("click", () => {
    const name = el("newProjectName")?.value.trim();
    if (!name) return showStatus("项目名称不能为空。", "warn");
    if (!state.projects[name]) {
      state.projects[name] = [];
      persistProjects(state);
    }
    el("newProjectName").value = "";
    el("projectName").value = name;
    renderProjects();
  });

  el("addDetector")?.addEventListener("click", () => {
    const project = el("projectName")?.value;
    const detector = el("newDetectorName")?.value.trim();
    if (!project) return showStatus("请先选择项目。", "warn");
    if (!detector) return showStatus("detector 名称不能为空。", "warn");
    const detectors = state.projects[project] || [];
    if (!detectors.includes(detector)) {
      detectors.push(detector);
      detectors.sort();
      state.projects[project] = detectors;
      persistProjects(state);
    }
    el("newDetectorName").value = "";
    renderProjects();
    el("detectorName").value = detector;
    syncPreview();
  });

  el("removeProject")?.addEventListener("click", () => {
    const project = el("projectName")?.value;
    if (!project) return;
    delete state.projects[project];
    persistProjects(state);
    renderProjects();
    showStatus(`已删除项目 ${project}。`, "info");
  });

  el("removeDetector")?.addEventListener("click", () => {
    const project = el("projectName")?.value;
    const detector = el("detectorName")?.value;
    if (!project || !detector) return;
    state.projects[project] = (state.projects[project] || []).filter((item) => item !== detector);
    persistProjects(state);
    renderProjects();
    showStatus(`已删除 detector ${detector}。`, "info");
  });

  const onInput = () => syncPreview();
  [
    "selfApiUrl",
    "workspaceRootDir",
    "originalDatasetDir",
    "trainEnv",
    "trainModel",
    "trainEpochs",
    "trainImgsz",
    "splitMode",
    "remoteHost",
    "remoteUsername",
    "remoteKeyPath",
    "remoteWorkspaceRootDir",
  ].forEach((id) => el(id)?.addEventListener("input", onInput));

  el("projectName")?.addEventListener("change", () => renderProjects());
  el("detectorName")?.addEventListener("change", () => {
    const p = el("projectName")?.value;
    if (p) renderProjectMeta(state, p);
    syncPreview();
  });

  document.querySelectorAll('input[name="runTarget"]').forEach((node) => {
    node.addEventListener("change", () => syncPreview());
  });

  el("launchButton")?.addEventListener("click", async () => {
    persistSettingsFromForm();

    const runTarget = getSelectedRunTarget();
    const fields = readFormFieldsForPayload(runTarget);
    const payload = buildTrainingWorkflowPayload(fields);

    if (!payload.workspace_root_dir) return showStatus("请在「设置」中填写 workspace_root_dir。", "warn");
    if (!payload.project_name) return showStatus("project_name 不能为空。", "warn");
    if (!payload.detector_name) return showStatus("detector_name 不能为空。", "warn");
    if (!payload.original_dataset_dir) return showStatus("未处理数据路径不能为空。", "warn");

    if (runTarget === "remote_sftp" && !hasFullSsh(fields)) {
      return showStatus("远端 SFTP 需在「设置」中填齐 remote_host、SSH 用户与私钥路径。", "warn");
    }

    persistRunTargetPref(runTarget);

    const launchBtn = el("launchButton");
    const prevLabel = launchBtn?.textContent ?? "";
    if (launchBtn) {
      launchBtn.disabled = true;
      launchBtn.setAttribute("aria-busy", "true");
      launchBtn.textContent = "提交中…";
    }

    showStatus("正在提交任务到 self_api，然后由后端转发到 n8n webhook。", "info");

    try {
      const resp = await fetch("/api/v1/tasks/launch-training-workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        credentials: "same-origin",
      });
      const data = await resp.json();
      if (!resp.ok) {
        const detail = data.detail;
        const msg = typeof detail === "string" ? detail : `HTTP ${resp.status}`;
        throw new Error(msg);
      }
      showStatus(`已提交。模式: ${data.workflow_mode}，上游状态码: ${data.upstream_status_code}`, "info");
    } catch (error) {
      showStatus(`提交失败: ${error.message}`, "error");
    } finally {
      if (launchBtn) {
        launchBtn.disabled = false;
        launchBtn.removeAttribute("aria-busy");
        launchBtn.textContent = prevLabel || "启动训练工作流";
      }
    }
  });

  el("copyPayload")?.addEventListener("click", () => {
    const text = el("payloadPreview")?.value ?? "";
    navigator.clipboard
      .writeText(text)
      .then(() => showStatus("任务体已复制。", "info"))
      .catch(() => showStatus("复制失败，请手动复制。", "warn"));
  });
}
