# Parameter Checklist

Use this checklist before routing the task.

Ask only for the fields that are still missing and that materially change the next action.

If the request is already obviously one-stage, do not stay in this skill. Route immediately.

## Common fields

- user goal
- current stage: raw data, cleaned dataset, YOLO dataset, ready-to-train dataset, or model file
- local path or remote path
- whether the task is local-only or cross-server

## Data preprocessing route

Collect these when routing to `$data-preprocess`:

- `input_dir` or `source_path`
- `output_dir`, `output_yaml_path`, or target location if output is required
- dataset layout:
  - flat `images+xmls`
  - nested leaf directories with mixed files
  - per-sample `images/` + `xmls/` subfolders
  - YOLO `images/` + `labels/`
- whether labels are XML or YOLO TXT
- whether the task is single-class
- whether transfer, zip, unzip, move, or copy is needed

## Model CLI route

Collect these when routing to `$ultralytics-yolo-modes`:

- mode: `train`, `val`, `predict`, or `export`
- dataset yaml path or model path
- target environment or execution location
- whether training is local or remote
- model weights
- output format if export is requested

For `train`, always collect:

- `project`
- `name`
- local-only: `project_root_dir`, `yolo_train_env`
- remote-only: `host`, `project_root_dir`, `yolo_train_env`, `username`, and SSH auth

## Combined workflow route

Collect the minimum fields for the next incomplete stage only.

Examples:

- If the dataset is still raw, start with preprocessing fields only.
- If preprocessing is done and only training remains, switch to model CLI fields.

## Routing heuristics

- Mentions of `clean`, `split`, `augment`, `crop`, `yaml`, `zip`, `sftp`, `upload`, `download`, `merge`, or `整理数据集` indicate `$data-preprocess`.
- Mentions of `train`, `val`, `predict`, `export`, `best.pt`, `data.yaml`, or `yolo` CLI indicate `$ultralytics-yolo-modes`.
- Mentions of both dataset prep and training indicate a combined workflow managed by this top-level skill.
- Mentions like `先整理数据再训练`, `从原始数据一路做到训练`, `给我完整 SOP`, or `长 SOP` indicate this skill.
- Mentions like `本地训练`, `远程训练`, `集群训练`, `sbatch`, or `提交任务到服务器` should force an explicit local-vs-remote training decision.
