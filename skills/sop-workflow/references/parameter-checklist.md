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

For `train`, always collect (MUST NOT silently default any of these):

- `detector_name`: stable model family name for directory structure
- `root_dir`: business project root directory
- `model`: ask for size preference (nano / small / medium / large / x) or custom checkpoint
- `batch`: ask the user; do not default silently
- `imgsz`: ask the user; remind that non-square sizes benefit strip or panoramic images
- `epochs`: or speed-vs-quality preference
- `project` and `name` (derived from rules once root_dir and detector_name are known)
- local-only: `project_root_dir`, `yolo_train_env`
- remote-only: `host`, `project_root_dir`, `yolo_train_env`, `username`, and SSH auth

## Dataset readiness check (MUST run before model CLI route for raw data)

When the user provides a raw dataset path (image folder, xml folder, or directory without YAML), always ask:

1. **Train/val split**: Has the dataset been split into train and val sets? If not, route to `$data-preprocess` split step.
2. **Image characteristics**: Are the images long horizontal strips or panoramic scenes?
   - If yes: strongly recommend sliding-window crop before training.
   - Explain: "Strip images trained at standard 640×640 will crop off context. Sliding-window crop produces small-image tiles that preserve spatial information for small-object detection."
   - Ask: enable sliding-window crop? If yes, collect: tile size, overlap ratio, output directory.
3. **Label format**: Are labels XML (VOC), JSON (COCO), or YOLO TXT? If XML or JSON, route to `$data-preprocess` for conversion first.
4. **Class list**: Confirm the class names and count if not evident from `classes.txt` or label files.

Do NOT proceed to training field collection until the user has confirmed or completed all applicable dataset-readiness steps.

## Combined workflow route

Collect the minimum fields for the next incomplete stage only.

Examples:

- If the dataset is still raw, start with dataset readiness checks, then preprocessing fields only.
- If preprocessing is done and only training remains, switch to model CLI fields.
- If images are long strips and split/sliding-window has not been done, complete those steps before any training parameter collection.

## Routing heuristics

- Mentions of `clean`, `split`, `augment`, `crop`, `yaml`, `zip`, `sftp`, `upload`, `download`, `merge`, or `整理数据集` indicate `$data-preprocess`.
- Mentions of `train`, `val`, `predict`, `export`, `best.pt`, `data.yaml`, or `yolo` CLI indicate `$ultralytics-yolo-modes`.
- Mentions of both dataset prep and training indicate a combined workflow managed by this top-level skill.
- Mentions like `先整理数据再训练`, `从原始数据一路做到训练`, `给我完整 SOP`, or `长 SOP` indicate this skill.
- Mentions like `本地训练`, `远程训练`, `集群训练`, `sbatch`, or `提交任务到服务器` should force an explicit local-vs-remote training decision.
- A raw image directory path or XML directory without a `.yaml` file always triggers dataset readiness checks before any training routing.
- Mention of `横向长条图`, `条带图像`, `全景图`, `strip image`, or `panoramic` always triggers a sliding-window crop recommendation.
