# Request Checklist

Use this file when the user gives partial information and wants fast execution.

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
- task type
- starting weights
- target epochs or speed-vs-quality preference
- enough information to derive `project`
- enough information to derive `name`

If required upstream fields are missing, ask the user before producing or running the final `train` CLI.

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
2. Resolve local file paths.
3. Produce one exact CLI.
4. Explain the artifact or directory that should appear after success.
5. If needed, follow with one tighter refinement command instead of many alternatives.
