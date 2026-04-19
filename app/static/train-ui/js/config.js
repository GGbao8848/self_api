/** 与训练页相关的本地存储键与默认值（与业务逻辑解耦）。 */

export const STORAGE_KEYS = {
  workspaces: "self_api_train_ui_workspaces",
  projects: "self_api_train_ui_projects",
  remote: "self_api_train_ui_remote",
  runTarget: "self_api_train_ui_run_target",
  /** 空字符串表示使用当前页面 origin */
  selfApiUrl: "self_api_train_ui_self_api_url",
};

/** UI 占位默认；可被用户覆盖并写入 localStorage */
export const DEFAULT_REMOTE_HOST = "172.31.42";

export const defaults = {
  workspaces: [],
  projects: {
    TVDS: ["nzxj_louyou"],
  },
  runTarget: "local",
};
