/** 纯展示与 DOM 更新，与提交、存储解耦 */

export function el(id) {
  return document.getElementById(id);
}

export function showStatus(message, kind = "info") {
  const box = el("statusBox");
  if (!box) return;
  box.className = `status show ${kind}`;
  box.textContent = message;
  // 错误对读屏更urgent；成功/进行用 status
  if (kind === "error") {
    box.setAttribute("role", "alert");
  } else {
    box.setAttribute("role", "status");
  }
}

export function renderWorkspaces(state, handlers) {
  const host = el("workspaceChips");
  if (!host) return;
  host.innerHTML = "";
  state.workspaces.forEach((workspace) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    const span = document.createElement("span");
    span.textContent = workspace;
    chip.appendChild(span);

    const useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.textContent = "用";
    useBtn.title = "使用这个工作区";
    useBtn.addEventListener("click", () => handlers.onUse(workspace));

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.textContent = "×";
    removeBtn.title = "删除";
    removeBtn.addEventListener("click", () => handlers.onRemove(workspace));

    chip.appendChild(useBtn);
    chip.appendChild(removeBtn);
    host.appendChild(chip);
  });
}

export function renderProjectMeta(state, currentProject) {
  const meta = el("projectMeta");
  if (!meta) return;
  const detectors = state.projects[currentProject] || [];
  meta.innerHTML = `
    <div class="meta-card">
      <strong>当前项目</strong>
      <div>${currentProject || "-"}</div>
    </div>
    <div class="meta-card">
      <strong>detector 列表</strong>
      <div>${detectors.length ? detectors.join(", ") : "暂无 detector"}</div>
    </div>
  `;
}

export function readFormFieldsForPayload(runTarget) {
  return {
    selfApiUrl: el("selfApiUrl")?.value,
    workspaceRootDir: el("workspaceRootDir")?.value,
    projectName: el("projectName")?.value,
    detectorName: el("detectorName")?.value,
    originalDatasetDir: el("originalDatasetDir")?.value,
    trainEnv: el("trainEnv")?.value,
    trainModel: el("trainModel")?.value,
    trainEpochs: el("trainEpochs")?.value,
    trainImgsz: el("trainImgsz")?.value,
    splitMode: el("splitMode")?.value,
    remoteHost: el("remoteHost")?.value,
    remoteUsername: el("remoteUsername")?.value,
    remoteKeyPath: el("remoteKeyPath")?.value,
    remoteWorkspaceRootDir: el("remoteWorkspaceRootDir")?.value,
    runTarget,
  };
}

export function getSelectedRunTarget() {
  const checked = document.querySelector('input[name="runTarget"]:checked');
  return checked?.value === "remote_sftp" ? "remote_sftp" : "local";
}

export function setRunTargetRadios(value) {
  const v = value === "remote_sftp" ? "remote_sftp" : "local";
  const radio = document.querySelector(`input[name="runTarget"][value="${v}"]`);
  if (radio) radio.checked = true;
}

export function updateCredentialHint(sshReady) {
  const hint = el("sshCredentialHint");
  if (!hint) return;
  hint.textContent = sshReady
    ? "SSH 已配齐：主界面可选「远端 SFTP」；仍可选「本地」——后端按 run_target 转发 n8n。"
    : "需 host、用户、私钥三项齐全才能选「远端 SFTP」；否则仅能本地。";
}

/** 主界面顶部摘要：便于确认当前 API / 路径（无需打开设置） */
export function updateRunSummaryFromPayload(payload) {
  const bar = el("runSummaryBar");
  if (!bar) return;
  const api = payload.self_api_url || "";
  const ws = payload.workspace_root_dir?.trim() ? payload.workspace_root_dir : "（未设置）";
  const rr =
    payload.remote_project_root_dir != null && String(payload.remote_project_root_dir).trim() !== ""
      ? payload.remote_project_root_dir
      : "—";
  bar.textContent = `self_api: ${api} · workspace: ${ws} · remote_workspace_root_dir: ${rr}`;
}
