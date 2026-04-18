---
name: ultralytics-yolo-modes
description: Use this skill when the user wants to train, validate, predict, or export an Ultralytics YOLO model and the task needs normalized training parameters, naming, project paths, and execution shape. For train requests in this self_api project, first decide whether execution is local or remote: local training injects the `yolo-train` API, remote training injects the `remote-sbatch-yolo-train` API. Direct raw CLI output is mainly for val, predict, export, or when the user explicitly asks for CLI only.
---

# Ultralytics YOLO Modes

Use this skill to normalize the training/inference/export stage for this project.

Compatibility baseline for this skill:

- target package behavior: `ultralytics v8.4.38`
- official docs used as the primary source on `2026-04-17`:
  - `https://docs.ultralytics.com/modes/train/`
  - `https://docs.ultralytics.com/modes/val/`
  - `https://docs.ultralytics.com/modes/predict/`
  - `https://docs.ultralytics.com/modes/export/`

If local installed CLI behavior differs from the docs, prefer the local CLI and state the discrepancy explicitly.

Map the request into one of four modes:

- `train`: fit a model on a dataset
- `val`: evaluate a trained or pretrained model
- `predict`: run inference on images, folders, videos, webcams, or streams
- `export`: convert a trained model into a deployment format

Read [references/modes.md](references/modes.md) first for mode routing.
Read [../sop-workflow/references/project-structure-kb.md](../sop-workflow/references/project-structure-kb.md) first for stable project directory and naming conventions.
Read [references/company-cli-standard.md](references/company-cli-standard.md) before generating the final output so company path, naming, and questioning rules are applied consistently.
Then load the matching mode reference for details:

- [references/train.md](references/train.md)
- [references/val.md](references/val.md)
- [references/predict.md](references/predict.md)
- [references/export.md](references/export.md)

Read [references/request-checklist.md](references/request-checklist.md) when the user request is underspecified.

## What to collect

Extract or infer these fields from the user request and local files:

- task: usually `detect`, `segment`, `classify`, `pose`, or `obb`
- mode: `train`, `val`, `predict`, or `export`
- `root_dir`: business project root directory
- `detector_name`: detector or model family name
- model input: pretrained weights like `yolo11n.pt` or custom weights like `runs/detect/train/weights/best.pt`
- data input: `dataset.yaml`, image path, video path, folder path, stream URL, or sample files
- execution target: local or remote
- goal: train best accuracy, quick smoke test, batch inference, ONNX export, TensorRT export, etc.
- runtime constraints: GPU/CPU, image size, batch size, latency target, export format

If the task type is not explicit, infer it from the dataset or labels when possible. If it is still ambiguous, ask one focused question instead of guessing.

## Workflow

1. Identify whether the user wants model creation, evaluation, inference, or deployment.
2. Resolve local or remote execution first.
3. Pick the YOLO task and mode.
4. Load [../sop-workflow/references/project-structure-kb.md](../sop-workflow/references/project-structure-kb.md) for shared project structure, then [references/company-cli-standard.md](references/company-cli-standard.md) for path and naming normalization.
5. For `train`, decide the API injection target:
   - local training -> `yolo-train`
   - remote training -> `remote-sbatch-yolo-train`
6. Load the matching mode reference and add only the arguments that materially affect the user's goal.
7. For `export`, follow the official CLI argument surface from the export docs. Do not invent generic `project=` or `name=` arguments, because export mode does not expose them as normal documented CLI parameters.
8. Tell the user what the final API payload or CLI will produce.

## Decision rules

### Train

Use `train` when the user has labeled data and wants a custom model.
Load [references/train.md](references/train.md) before drafting the final command.
For this project, `train` is not one generic output path:

- local train requests inject `POST /api/v1/preprocess/yolo-train`
- remote train requests inject `POST /api/v1/preprocess/remote-sbatch-yolo-train`

### Val

Use `val` when the user wants mAP, precision, recall, speed, or a checkpoint comparison.
Load [references/val.md](references/val.md) before drafting the final command.

### Predict

Use `predict` when the user wants inference outputs from media or streams.
Load [references/predict.md](references/predict.md) before drafting the final command.

### Export

Use `export` when the user needs deployment artifacts or deployable converted models.
Load [references/export.md](references/export.md) before drafting the final command.

## Response style

When acting on a request, keep the response operational:

- first, state the inferred task, mode, and execution target
- then:
  - for local train, provide the `yolo-train` API payload shape
  - for remote train, provide the `remote-sbatch-yolo-train` API payload shape
  - for val/predict/export, provide the exact CLI
- then, mention expected outputs and the next verification step

If the user has already provided enough local context, execute the appropriate API or command instead of stopping at advice.

## Trigger examples

Use this skill for requests like:

- `我已经有 dataset.yaml，帮我整理本地训练参数`
- `这个训练要跑远程集群，帮我整理成 sbatch 提交参数`
- `用 best.pt 帮我生成 val 命令`
- `把这个模型导出成 onnx`
- `数据已经整理好了，只需要训练阶段参数`

Do not use this skill for requests like:

- `先帮我把原始数据整理好`
- `把多层目录 xml 数据清洗汇总`
