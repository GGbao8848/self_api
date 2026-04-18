# Company CLI Standard

Use this file as the final normalization layer before generating any Ultralytics YOLO CLI for this company.

Compatibility baseline: `ultralytics v8.4.38`

This file defines:

- required path structure
- naming rules
- mandatory questions
- mode bucket mapping
- safe defaults

If any mode-specific reference conflicts with this file, follow this file for company-standard CLI generation, except where the official Ultralytics CLI does not expose the company-preferred argument shape. Export mode is the main exception: the final export CLI must stay within the official documented export argument surface.

## Required fields to collect

Before producing a final CLI, collect or infer:

- `mode`: `train`, `val`, `predict`, or `export`
- `task`: usually `detect`, `segment`, `classify`, `pose`, or `obb`
- `root_dir`: business project root directory
- `detector_name`: detector or model family name
- `model`: starting weights or source model path
- input target:
  - `data` for `train` and most `val`
  - `source` for `predict`
  - `format` for `export`

Additional required field for `train`, `val`, and `predict`:

- `name`: run or version name

For `export`, keep an inferred version label when helpful for explanation, but do not require it as a CLI argument.

## Naming semantics

Two names exist and they must not be confused:

- `detector_name`
  - the detector itself
  - the stable model family name
  - used in the project path
- `name`
  - the current detector version or run version
  - used as the leaf run directory

For training, the preferred rule is:

- dataset YAML filename should usually be `detector_name + timestamp`
- `name` must equal the dataset YAML filename stem

Example:

- `detector_name=name1`
- dataset YAML: `name1_20260417.yaml`
- final `name=name1_20260417`

This keeps:

- dataset version
- weight folder
- training run name

aligned to the same version string.

## Fixed run-bucket dictionary

Use only these bucket names:

- `train`
  - `detect -> runs/detect`
  - `segment -> runs/segment`
  - `classify -> runs/classify`
  - `pose -> runs/pose`
  - `obb -> runs/obb`
- `val -> runs/val`
- `predict -> runs/predict`
- `export -> runs/export`

Never invent mixed bucket names such as:

- `runs/detect_predict`
- `runs/detect_val`
- `runs/detect_export`

## Project path formula

Build `project` as:

```text
<root_dir>/<detector_name>/<bucket>
```

Where `<bucket>` comes from the fixed run-bucket dictionary.

Then the final run directory becomes:

```text
<project>/<name>
```

Examples:

- detection train:
  - `project=/mnt/usrhome/sk/ndata/TVDS/name1/runs/detect`
  - `name=name1_20260417`
  - final run dir:
    - `/mnt/usrhome/sk/ndata/TVDS/name1/runs/detect/name1_20260417`
- predict:
  - `project=/mnt/usrhome/sk/ndata/TVDS/name1/runs/predict`
  - `name=name1_20260417`
- val:
  - `project=/mnt/usrhome/sk/ndata/TVDS/name1/runs/val`
  - `name=name1_20260417`
- export:
  - keep the inferred version name for description and bookkeeping
  - do not force `project` or `name` into the CLI unless the chosen export format explicitly supports them

## Mandatory questioning rules

Do not generate the final CLI until these are clear.

Core rule:

- prefer asking over guessing whenever a required field for company-standard CLI generation is missing
- if the required upstream fields are complete, derive `project` and `name` from the company rules for `train`, `val`, and `predict` instead of asking the user to spell them out again
- for `export`, derive only what the official export CLI actually accepts and describe the artifact path separately

### For train

Must know:

- `root_dir`
- `detector_name`
- dataset YAML path
- `task`
- `model`

Then enforce:

- `project` derived from `root_dir + detector_name + runs/<task-bucket>`
- `name` equals dataset YAML stem

Ask the user if any of these are missing.

Also ask for confirmation if:

- dataset YAML stem does not start with `detector_name`
- task cannot be inferred reliably

### For val

Must know:

- `root_dir`
- `detector_name`
- `model`
- dataset YAML path when required

Ask the user if any of these are missing.

Preferred version naming:

- if validating a training run, reuse that run's version name as `name`
- otherwise ask the user for the validation run name

### For predict

Must know:

- `root_dir`
- `detector_name`
- `model`
- `source`

Ask the user if any of these are missing.

Preferred version naming:

- if using a known trained weights file, reuse the corresponding version name as `name`
- otherwise ask the user for the desired prediction output name

### For export

Must know:

- `root_dir`
- `detector_name`
- `model`
- `format`

Ask the user if any of these are missing.

Preferred version naming:

- if exporting a known trained weights file, reuse the corresponding version name as `name`
- otherwise ask the user for the export version label only if it helps describe the expected artifact; do not force it into the CLI

## Safe defaults

Use these defaults unless the user request or environment suggests otherwise.

- for `predict` and `export`, prefer the training `imgsz` recorded in the sibling `args.yaml` next to the weights directory
- if `args.yaml` is unavailable or unreadable, fall back to `imgsz=640`
- if reading or reusing training `imgsz` causes shape or tensor mismatch style errors, stop and ask the user before forcing another `imgsz`
- lightweight pretrained checkpoint for smoke tests
- `device=0` only when GPU use is intended and plausible
- `save=True` for predict when the user expects visible outputs
- `save_txt=True` only when labels are explicitly useful

Training-specific defaults:

- keep pretrained initialization unless the user explicitly wants from-scratch training
- use `name=<dataset_yaml_stem>`
- use `project=<root_dir>/<detector_name>/runs/<task-bucket>`

Inference defaults:

- when the model path is something like:
  - `<root_dir>/<detector_name>/runs/detect/<name>/weights/best.pt`
- first check:
  - `<root_dir>/<detector_name>/runs/detect/<name>/args.yaml`
- if `args.yaml` contains `imgsz`, reuse that value in `predict` and `export`
- if not, use `imgsz=640`

Export defaults:

- if `args.yaml` contains `imgsz`, reuse that value in `export`
- if not, use `imgsz=640`
- default `format=torchscript`
- default `half=True`
- exported files usually appear adjacent to the source checkpoint or in a format-specific sibling directory

## Standard CLI patterns

### Train

```bash
yolo TASK train model=MODEL data=DATA_YAML project=<root_dir>/<detector_name>/runs/TASK_BUCKET name=<yaml_stem> imgsz=640
```

### Val

```bash
yolo TASK val model=MODEL data=DATA_YAML project=<root_dir>/<detector_name>/runs/val name=VERSION_NAME imgsz=640
```

### Predict

```bash
yolo TASK predict model=MODEL source=SOURCE project=<root_dir>/<detector_name>/runs/predict name=VERSION_NAME imgsz=TRAIN_IMGSZ
```

### Export

```bash
yolo export model=MODEL format=torchscript imgsz=TRAIN_IMGSZ half=True
```

## Final company-standard examples

Use these as final-form templates when the required fields are already known.

### 1. Train

Scenario:

- `root_dir=/mnt/usrhome/sk/ndata/TVDS`
- `detector_name=name1`
- dataset YAML: `/data/name1_20260417.yaml`
- `yaml_stem=name1_20260417`

```bash
yolo detect train model=yolo11n.pt data=/data/name1_20260417.yaml project=/mnt/usrhome/sk/ndata/TVDS/name1/runs/detect name=name1_20260417 imgsz=640
```

### 2. Val

Scenario:

- `root_dir=/mnt/usrhome/sk/ndata/TVDS`
- `detector_name=name1`
- version name: `name1_20260417`

```bash
yolo detect val model=/mnt/usrhome/sk/ndata/TVDS/name1/runs/detect/name1_20260417/weights/best.pt data=/data/name1_20260417.yaml project=/mnt/usrhome/sk/ndata/TVDS/name1/runs/val name=name1_20260417 imgsz=640
```

### 3. Predict

Scenario:

- `root_dir=/mnt/usrhome/sk/ndata/TVDS`
- `detector_name=name1`
- version name: `name1_20260417`
- inference source: `/data/demo.jpg`

```bash
yolo detect predict model=/mnt/usrhome/sk/ndata/TVDS/name1/runs/detect/name1_20260417/weights/best.pt source=/data/demo.jpg project=/mnt/usrhome/sk/ndata/TVDS/name1/runs/predict name=name1_20260417 imgsz=1280 save=True
```

### 4. Export

Scenario:

- `root_dir=/mnt/usrhome/sk/ndata/TVDS`
- `detector_name=name1`
- version name: `name1_20260417`

```bash
yolo export model=/mnt/usrhome/sk/ndata/TVDS/name1/runs/detect/name1_20260417/weights/best.pt format=torchscript imgsz=1280 half=True
```

In the example above, `imgsz=1280` is assumed to have been read from:

```text
/mnt/usrhome/sk/ndata/TVDS/name1/runs/detect/name1_20260417/args.yaml
```

## Resume rule

Do not use generic default paths like `runs/TASK/train/...`.

Resume should follow the company path convention:

```bash
yolo train resume model=<root_dir>/<detector_name>/runs/<task-bucket>/<name>/weights/last.pt
```

## Final output checklist

Before returning a final CLI, verify:

- bucket name matches the fixed dictionary
- for `train`, `val`, and `predict`, `project` uses `<root_dir>/<detector_name>/<bucket>`
- for `train`, `val`, and `predict`, `name` is present
- for train, `name` equals dataset YAML stem
- for predict and export, `imgsz` should first try to match the training `args.yaml`
- for export, the CLI contains only officially documented export arguments for the chosen format
- if tensor or shape mismatch errors appear after reusing training `imgsz`, ask the user before changing it
- no mixed bucket names are used
- command includes only arguments that materially help the stated goal

## Prohibited outputs

Do not generate a final company-standard CLI if any of these are true:

- missing `root_dir` or `detector_name` while still guessing a company path for `train`, `val`, or `predict`
- missing `name` for `train`, `val`, or `predict`
- for train, `name` does not match the dataset YAML filename stem
- for predict or export, `imgsz` is guessed without first checking the sibling `args.yaml` when that path should exist
- changing `imgsz` after tensor or shape mismatch style errors without asking the user
- using mixed bucket names such as `runs/detect_predict`, `runs/detect_val`, or `runs/detect_export`
- using generic default output paths such as `runs/train*`, `runs/predict*`, `runs/val*`, or `runs/export*` instead of the company path formula
- adding unsupported export arguments such as generic `project=` or `name=` to `yolo export ...`
