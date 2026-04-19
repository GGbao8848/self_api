# Request Checklist

Use this file when the user gives partial information and wants fast execution.

For `train`, the default result in this project is an API payload, not a raw CLI, unless the user explicitly asks for CLI only.

For company path, naming, bucket, default, and prohibited-output rules, follow [company-cli-standard.md](company-cli-standard.md). This file is only a fast field-gathering aid.

## STOP rule for train

**NEVER produce a train API payload or CLI before all mandatory fields are confirmed.**

If any mandatory field is missing, send one message that collects all missing fields at once. Do not drip-feed one question at a time, and do not guess silently.

## STOP rule — preview before any side effect (all modes)

**NEVER call a preprocess API, never run `curl` to the API, and never start `yolo train` in the first assistant reply** — even if the user said `用 skill 训练`, `本地训练`, or `虚拟环境 xxx`.

First reply must be only:

- missing-field questions, and/or
- a **full preview** (endpoints or CLI in order, every parameter with meaning, every output path), ending with "确认后执行".

Execution belongs to a **later** turn, after the user explicitly confirms the preview for **this** conversation.

## Fast classification

Map the request into one sentence:

- "I have labeled data and want a usable model" -> `train`
- "I have a model and want metrics" -> `val`
- "I have a model and want outputs on media" -> `predict`
- "I have a model and want deployable files" -> `export`

## Missing fields to infer or ask for

### For train

**Dataset readiness check (ask first)**

Before collecting training parameters, check what the user actually provided:

- If the user provides a raw directory (images, xmls, annotations, or mixed files), it is NOT a ready dataset.
  - Ask: Has the dataset been split into train/val sets?
  - Ask: Are the images long horizontal strip or panoramic scenes? (If yes, recommend sliding-window crop and ask whether to enable it before preprocessing.)
  - Ask: Do the labels need to be converted from XML/JSON to YOLO TXT format?
  - Route to `$data-preprocess` or `$sop-workflow` for dataset preparation before proceeding.
- Only continue to training fields once the user has (or will have) a valid dataset YAML.

**Training fields (must collect all before generating output)**

- `detector_name`: stable model family name; used in directory structure; MUST ask if not given, do NOT infer
- `root_dir`: business project root directory where runs will be stored; MUST ask if not given, do NOT infer
- dataset YAML path: must be a `.yaml` file with `train:`, `val:`, `nc:` fields
- execution target: local or remote; MUST ask if not given, do NOT infer
- task type: detect, segment, classify, pose, or obb; infer from dataset if possible, otherwise ask
- `model`: starting weights; ask for size preference (nano / small / medium / large / x) or custom `.pt` path; do NOT silently default to any size
- `batch`: training batch size; ask the user; do NOT silently default
- `imgsz`: training image size; ask the user; remind that strip or panoramic images often benefit from non-square sizes like 1280×320 or 1920×384
- target `epochs` or speed-vs-quality preference
- local train: `project_root_dir`, `yolo_train_env`
- remote train: `host`, `project_root_dir`, `yolo_train_env`, `username`, and SSH auth

**Pre-execution confirmation**

After all fields are collected, show a summary table with all parameters and explain what each will produce:
- example: "batch=16 + imgsz=640 will fit approximately 8 GB GPU memory and run for ~N hours at 100 epochs"
- example: "model=yolo11s.pt gives a balance of speed and accuracy for small-object detection"

Ask the user to confirm before calling any API or running any command.

If required upstream fields are missing, ask the user before producing or running the final train output.

Default train output:

- local -> `yolo-train` API payload
- remote -> `remote-sbatch-yolo-train` API payload

CLI-only exception:

- if the user explicitly asks for `只要 CLI`, `不要 API`, `原生命令`, `just the command`, or equivalent, return raw `yolo train ...`

### For val

- `root_dir`
- `detector_name`
- weights path
- dataset YAML path
- whether the user cares about speed, mAP, or both
- enough information to derive `project`
- enough information to derive `name`

For company-standard validation:

- use `project=<root_dir>/<detector_name>/runs/val`
- reuse the training version name as `name` when it can be inferred from the model path
- otherwise ask the user for the validation run name

### For predict

- `root_dir`
- `detector_name`
- weights path
- source path or stream
- whether to save images, labels, confidences, or all of them
- enough information to derive `project`
- enough information to derive `name`
- `imgsz`

For company-standard prediction:

- use `project=<root_dir>/<detector_name>/runs/predict`
- reuse the training version name as `name` when it can be inferred from the model path
- resolve `imgsz` from the sibling `args.yaml` first
- if `args.yaml` is missing or unreadable, fall back to `640`
- if reusing that `imgsz` leads to tensor or shape mismatch style errors, ask the user before changing it

### For export

- `root_dir`
- `detector_name`
- weights path
- target format, default `torchscript`
- deployment environment such as CPU, GPU, TensorRT, mobile, or browser-adjacent runtime
- optional version label if it helps describe the expected artifact
- `imgsz`
- `half`

For company-standard export:

- default `format=torchscript`
- default `half=True`
- resolve `imgsz` from the sibling `args.yaml` first
- if `args.yaml` is missing or unreadable, fall back to `640`
- if reusing that `imgsz` leads to tensor or shape mismatch style errors, ask the user before changing it
- do not add generic `project=` or `name=` to `yolo export ...` because they are not standard export CLI parameters in the official docs
- only include format-specific extras that are documented for the chosen backend

## Good completion pattern

1. Infer task and mode.
2. For `train`, resolve whether execution is local or remote first.
3. For `train`, produce one normalized API payload by default.
4. For CLI-only exceptions or non-train modes, produce one exact CLI.
5. Explain the artifact or directory that should appear after success.
6. If needed, follow with one tighter refinement instead of many alternatives.
