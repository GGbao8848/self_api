# Request Checklist

Use this file when the user gives partial information and wants fast execution.

For `train`, the default result in this project is an API payload, not a raw CLI, unless the user explicitly asks for CLI only.

For company path, naming, bucket, default, and prohibited-output rules, follow [company-cli-standard.md](company-cli-standard.md). This file is only a fast field-gathering aid.

## Fast classification

Map the request into one sentence:

- "I have labeled data and want a usable model" -> `train`
- "I have a model and want metrics" -> `val`
- "I have a model and want outputs on media" -> `predict`
- "I have a model and want deployable files" -> `export`

## Missing fields to infer or ask for

### For train

- `root_dir`
- `detector_name`
- dataset YAML path
- whether execution is local or remote
- task type
- starting weights
- target epochs or speed-vs-quality preference
- enough information to derive `project`
- enough information to derive `name`
- local train: `project_root_dir`, `yolo_train_env`
- remote train: `host`, `project_root_dir`, `yolo_train_env`, `username`, and SSH auth

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
