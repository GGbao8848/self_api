/**
 * 组装提交给 `/api/v1/tasks/launch-training-workflow` 的请求体。
 * DOM 读取由调用方传入，本模块不依赖具体 id，便于测试与替换 UI。
 */

/**
 * @param {object} fields
 * @param {string} fields.workspaceRootDir
 * @param {string} fields.projectName
 * @param {string} fields.detectorName
 * @param {string} fields.originalDatasetDir
 * @param {string} fields.trainEnv
 * @param {string} fields.trainModel
 * @param {string|number} fields.trainEpochs
 * @param {string|number} fields.trainImgsz
 * @param {string} fields.splitMode
 * @param {string} fields.remoteHost
 * @param {string} fields.remoteUsername
 * @param {string} fields.remoteKeyPath
 * @param {string} fields.remoteWorkspaceRootDir
 * @param {string} [fields.selfApiUrl]
 * @param {"local"|"remote_sftp"} fields.runTarget
 */
export function buildTrainingWorkflowPayload(fields) {
  const trim = (s) => (s || "").trim();
  const fallbackOrigin = typeof window !== "undefined" ? window.location.origin : "";
  const selfApi = trim(fields.selfApiUrl) || fallbackOrigin;
  return {
    self_api_url: selfApi,
    workspace_root_dir: trim(fields.workspaceRootDir),
    project_name: fields.projectName || "",
    detector_name: fields.detectorName || "",
    original_dataset_dir: trim(fields.originalDatasetDir),
    yolo_train_env: trim(fields.trainEnv) || "yolo_pose",
    yolo_train_model: trim(fields.trainModel) || "yolo11s.pt",
    yolo_train_epochs: Number(fields.trainEpochs || 5),
    yolo_train_imgsz: Number(fields.trainImgsz || 640),
    split_mode: fields.splitMode || "train_val",
    run_target: fields.runTarget,
    remote_host: trim(fields.remoteHost) || null,
    remote_username: trim(fields.remoteUsername) || null,
    remote_private_key_path: trim(fields.remoteKeyPath) || null,
    remote_project_root_dir: trim(fields.remoteWorkspaceRootDir) || null,
  };
}

/** 三项核心 SSH 字段是否齐全（与后端「可远端」判定一致）。 */
export function hasFullSsh(fields) {
  const t = (s) => (s || "").trim();
  return Boolean(t(fields.remoteHost) && t(fields.remoteUsername) && t(fields.remoteKeyPath));
}
