---
name: ultralytics-yolo-modes
description: Use this skill when the user wants to train, validate, predict, or export an Ultralytics YOLO model with CLI commands. Trigger it when the user provides datasets, images, videos, model files, deployment targets, or a computer-vision goal such as custom training, evaluation, inference, or model conversion, and wants the agent to quickly choose the right YOLO mode and produce or run the correct `yolo` CLI.
---

# Ultralytics YOLO Modes

Use this skill to turn a user's data and goal into the shortest reliable Ultralytics YOLO CLI workflow.

Compatibility baseline for this skill:

- target package behavior: `ultralytics v8.4.38`
- official docs used as the primary source on `2026-04-17`:
  - `https://docs.ultralytics.com/modes/train/`
  - `https://docs.ultralytics.com/modes/val/`
  - `https://docs.ultralytics.com/modes/predict/`
  - `https://docs.ultralytics.com/modes/export/`

If local installed CLI behavior differs from the docs, prefer the local CLI and state the discrepancy explicitly.

Prefer official Ultralytics CLI syntax and map the request into one of four modes:

- `train`: fit a model on a dataset
- `val`: evaluate a trained or pretrained model
- `predict`: run inference on images, folders, videos, webcams, or streams
- `export`: convert a trained model into a deployment format

Read [references/modes.md](references/modes.md) first for mode routing.
Read [references/company-cli-standard.md](references/company-cli-standard.md) before generating the final CLI so company path, naming, and questioning rules are applied consistently.
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
- goal: train best accuracy, quick smoke test, batch inference, ONNX export, TensorRT export, etc.
- runtime constraints: GPU/CPU, image size, batch size, latency target, export format

If the task type is not explicit, infer it from the dataset or labels when possible. If it is still ambiguous, ask one focused question instead of guessing.

## Workflow

1. Identify whether the user wants model creation, evaluation, inference, or deployment.
2. Resolve local paths before writing a command.
3. Pick the YOLO task and mode.
4. Load [references/company-cli-standard.md](references/company-cli-standard.md) as the single source of truth for company path, naming, bucket, questioning, default, and prohibited-output rules.
5. Load the matching mode reference and add only the arguments that materially affect the user's goal.
6. For `export`, follow the official CLI argument surface from the export docs. Do not invent generic `project=` or `name=` arguments, because export mode does not expose them as normal documented CLI parameters.
7. Tell the user what the command will produce. For `train`, `val`, and `predict`, this is usually the company-standard run directory. For `export`, describe the expected exported artifact path or directory pattern instead of forcing a company run directory.

## Decision rules

### Train

Use `train` when the user has labeled data and wants a custom model.
Load [references/train.md](references/train.md) before drafting the final command.

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

- first, state the inferred task and mode
- then, provide the exact CLI
- then, mention expected outputs and the next verification step

If the user has already provided enough local context, execute the command instead of stopping at advice.
