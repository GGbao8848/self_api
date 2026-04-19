import { STORAGE_KEYS, defaults } from "./config.js";
import { loadJson, saveJson } from "./lib/jsonStorage.js";

/** 内存态 + 持久化：工作区、项目/detector、远端 SSH 备忘、本次执行目标偏好 */
export function createTrainUiState() {
  return {
    workspaces: loadJson(STORAGE_KEYS.workspaces, defaults.workspaces),
    projects: loadJson(STORAGE_KEYS.projects, defaults.projects),
    remote: loadJson(STORAGE_KEYS.remote, {}),
    runTargetPref: loadJson(STORAGE_KEYS.runTarget, defaults.runTarget),
    /** 自定义 API 根 URL（不含路径）；persist 为空串表示跟随页面 origin */
    selfApiUrlStored: loadJson(STORAGE_KEYS.selfApiUrl, ""),
  };
}

export function persistSelfApiUrl(value) {
  saveJson(STORAGE_KEYS.selfApiUrl, value ?? "");
}

export function persistWorkspaces(state) {
  saveJson(STORAGE_KEYS.workspaces, state.workspaces);
}

export function persistProjects(state) {
  saveJson(STORAGE_KEYS.projects, state.projects);
}

export function persistRemoteSnapshot(remote) {
  saveJson(STORAGE_KEYS.remote, remote);
}

export function persistRunTargetPref(value) {
  saveJson(STORAGE_KEYS.runTarget, value);
}
