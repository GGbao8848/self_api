# Company CLI Standard

Upstream project-structure knowledge lives in:

- `../sop-workflow/references/project-structure-kb.md`

Use this file as the final normalization layer before generating any Ultralytics YOLO training payload or Ultralytics CLI for this company.

Compatibility baseline: `ultralytics v8.4.38`

This file defines:

- required path structure
- naming rules
- dataset-to-run relationship
- mandatory questions
- mode bucket mapping
- safe defaults

If any mode-specific reference conflicts with this file, follow this file for company-standard output generation. For this self_api project, `train` defaults to normalized API payload generation, while `val`, `predict`, and `export` still default to final CLI generation. Export mode remains constrained to the official documented export argument surface.

If this file and the shared project-structure KB overlap, treat the KB as the macro directory convention and this file as the execution-shape normalization layer.

Dataset location should normally follow the KB rule:

```text
<root_dir>/<detector_name>/datasets/<dataset_version>
```

For training, `name` should usually align with the dataset version name so dataset version, YAML stem, and run directory stay aligned.

Hard rule:

- `<dataset_version>` must equal the dataset YAML filename stem

Preferred alignment:

- dataset folder name = dataset version
- dataset YAML stem = dataset version
- train `name` = dataset version

## Required fields to collect

Before producing a final output, collect or infer:

- `mode`: `train`, `val`, `predict`, or `export`
- `task`: usually `detect`, `segment`, `classify`, `pose`, or `obb`
- `root_dir`: business project root directory
- `detector_name`: detector or model family name
- `model`: starting weights or source model path
- execution target for `train`: local or remote
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

- dataset YAML filename should usually be `detector_name + YYYYMMDD_HHMM`
- `name` must equal the dataset YAML filename stem

Example:

- `detector_name=nzxj_louyou`
- dataset YAML: `nzxj_louyou_20260417_1430.yaml`
- final `name=nzxj_louyou_20260417_1430`

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
  - `project=/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect`
  - `name=nzxj_louyou_20260417_1430`
  - final run dir:
    - `/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect/nzxj_louyou_20260417_1430`
- predict:
  - `project=/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/predict`
  - `name=nzxj_louyou_20260417_1430`
- val:
  - `project=/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/val`
  - `name=nzxj_louyou_20260417_1430`
- export:
  - keep the inferred version name for description and bookkeeping
  - do not force `project` or `name` into the CLI unless the chosen export format explicitly supports them

## Mandatory questioning rules

Do not generate the final train payload or final CLI until these are clear.

Core rule:

- prefer asking over guessing whenever a required field for company-standard output generation is missing
- NEVER silently default `batch`, `imgsz`, or `model` size without asking the user
- if the required upstream fields are complete, derive `project` and `name` from the company rules for `train`, `val`, and `predict` instead of asking the user to spell them out again
- for `export`, derive only what the official export CLI actually accepts and describe the artifact path separately

### For train

**Phase 1 — Dataset readiness (MUST complete before Phase 2)**

Check whether the input is a ready dataset YAML or a raw data path:

- If the user provides an image folder, XML folder, or any directory without a `.yaml` file:
  - STOP. Ask the following before any training field collection:
    1. Has the dataset been split into train/val sets?
    2. Are the images long horizontal strips or panoramic scenes? If yes, sliding-window crop is recommended — ask whether to enable it.
    3. Are labels in XML/JSON format and need conversion to YOLO TXT?
  - Route to `$data-preprocess` or `$sop-workflow` to complete data preparation.
  - Do NOT proceed to Phase 2 until a valid dataset YAML is confirmed or will be produced.

**Phase 2 — Training parameter collection (collect all in one message)**

Must know:

- execution target: `local` or `remote`; MUST ask if ambiguous, never infer
- `root_dir`: business project root; MUST ask if not given, never infer from file paths
- `detector_name`: stable model family name; MUST ask if not given, never infer from dataset name
- dataset YAML path: must be a confirmed `.yaml` file
- `task`: infer from dataset labels if possible, otherwise ask
- `model`: ask for size preference (nano/small/medium/large/x) or custom checkpoint path; never silently default
- `batch`: ask the user; never silently default
- `imgsz`: ask the user; note that strip/panoramic images often benefit from non-square sizes such as 1280×320 or 1920×384
- `epochs`: ask or confirm a speed-vs-quality preference

Then enforce:

- `project` derived from `root_dir + detector_name + runs/<task-bucket>`
- `name` equals dataset YAML stem

**Phase 3 — Pre-execution confirmation**

Before calling any API or command, show the user a parameter summary:

```
detector_name : <value>
root_dir      : <value>
model         : <value>  → <what this means for speed/accuracy>
batch         : <value>  → <GPU memory implication>
imgsz         : <value>  → <resolution trade-off>
epochs        : <value>
execution     : local / remote
project       : <derived value>
name          : <derived value>
```

Explain consequences for each parameter. For example:
- "batch=16 + imgsz=640 will fit on a single 8 GB GPU and run for ~N hours at 100 epochs"
- "yolo11s.pt balances speed and accuracy for small-object detection"

Ask the user to confirm. Only proceed after explicit confirmation.

Output rule:

- if execution target is `local`, return or run the normalized `POST /api/v1/preprocess/yolo-train` payload
- if execution target is `remote`, return or run the normalized `POST /api/v1/preprocess/remote-sbatch-yolo-train` payload
- only return raw `yolo train ...` when the user explicitly says they do not want API execution, asks for command only, or uses equivalent wording such as `只给我命令`, `不要 API`, `原生命令`, `只要 cli`, or `just the command`

Also ask for confirmation if:

- dataset YAML stem does not start with `detector_name`
- task cannot be inferred reliably
- execution target is still ambiguous

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

## Standard output patterns

### Train API payloads

Local:

```json
{
  "model": "MODEL",
  "yaml_path": "DATA_YAML",
  "project": "<root_dir>/<detector_name>/runs/TASK_BUCKET",
  "name": "<yaml_stem>",
  "task": "TASK",
  "imgsz": 640
}
```

Remote:

```json
{
  "host": "REMOTE_HOST",
  "model": "MODEL",
  "yaml_path": "DATA_YAML",
  "project_root_dir": "<root_dir>",
  "project": "<root_dir>/<detector_name>/runs/TASK_BUCKET",
  "name": "<yaml_stem>",
  "task": "TASK",
  "imgsz": 640
}
```

CLI-only exception:

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
- `detector_name=nzxj_louyou`
- dataset YAML: `/data/nzxj_louyou_20260417_1430.yaml`
- `yaml_stem=nzxj_louyou_20260417_1430`
- execution target: `local`

```json
{
  "model": "yolo11n.pt",
  "yaml_path": "/data/nzxj_louyou_20260417_1430.yaml",
  "project": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect",
  "name": "nzxj_louyou_20260417_1430",
  "task": "detect",
  "imgsz": 640
}
```

### 1b. Train, remote

Scenario:

- `root_dir=/mnt/usrhome/sk/ndata/TVDS`
- `detector_name=nzxj_louyou`
- dataset YAML: `/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml`
- `yaml_stem=nzxj_louyou_20260417_1430`
- execution target: `remote`

```json
{
  "host": "gpu-cluster-01",
  "model": "yolo11n.pt",
  "yaml_path": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml",
  "project_root_dir": "/mnt/usrhome/sk/ndata/TVDS",
  "project": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect",
  "name": "nzxj_louyou_20260417_1430",
  "task": "detect",
  "imgsz": 640
}
```

### 1c. Train, CLI-only exception

Only when the user explicitly asks for raw command output:

```bash
yolo detect train model=yolo11n.pt data=/data/nzxj_louyou_20260417_1430.yaml project=/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect name=nzxj_louyou_20260417_1430 imgsz=640
```

### 2. Val

Scenario:

- `root_dir=/mnt/usrhome/sk/ndata/TVDS`
- `detector_name=nzxj_louyou`
- version name: `nzxj_louyou_20260417_1430`

```bash
yolo detect val model=/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect/nzxj_louyou_20260417_1430/weights/best.pt data=/data/nzxj_louyou_20260417_1430.yaml project=/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/val name=nzxj_louyou_20260417_1430 imgsz=640
```

### 3. Predict

Scenario:

- `root_dir=/mnt/usrhome/sk/ndata/TVDS`
- `detector_name=nzxj_louyou`
- version name: `nzxj_louyou_20260417_1430`
- inference source: `/data/demo.jpg`

```bash
yolo detect predict model=/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect/nzxj_louyou_20260417_1430/weights/best.pt source=/data/demo.jpg project=/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/predict name=nzxj_louyou_20260417_1430 imgsz=1280 save=True
```

### 4. Export

Scenario:

- `root_dir=/mnt/usrhome/sk/ndata/TVDS`
- `detector_name=nzxj_louyou`
- version name: `nzxj_louyou_20260417_1430`

```bash
yolo export model=/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect/nzxj_louyou_20260417_1430/weights/best.pt format=torchscript imgsz=1280 half=True
```

In the example above, `imgsz=1280` is assumed to have been read from:

```text
/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect/nzxj_louyou_20260417_1430/args.yaml
```

## Resume rule

Do not use generic default paths like `runs/TASK/train/...`.

Resume should follow the company path convention:

```bash
yolo train resume model=<root_dir>/<detector_name>/runs/<task-bucket>/<name>/weights/last.pt
```

## Final output checklist

Before returning a final output, verify:

- bucket name matches the fixed dictionary
- for `train`, `val`, and `predict`, `project` uses `<root_dir>/<detector_name>/<bucket>`
- for `train`, `val`, and `predict`, `name` is present
- for train, `name` equals dataset YAML stem
- for train, execution target is explicit: `local` or `remote`
- for local train, the default output is the `yolo-train` API payload
- for remote train, the default output is the `remote-sbatch-yolo-train` API payload
- for raw train CLI, confirm the user explicitly asked for command-only output
- for predict and export, `imgsz` should first try to match the training `args.yaml`
- for export, the CLI contains only officially documented export arguments for the chosen format
- if tensor or shape mismatch errors appear after reusing training `imgsz`, ask the user before changing it
- no mixed bucket names are used
- command includes only arguments that materially help the stated goal

## Prohibited outputs

Do not generate a final company-standard output if any of these are true:

- missing `root_dir` or `detector_name` while still guessing a company path for `train`, `val`, or `predict`
- missing `name` for `train`, `val`, or `predict`
- for train, missing execution target while still choosing local API, remote API, or CLI-only output
- for train, `name` does not match the dataset YAML filename stem
- for train, returning raw `yolo train ...` without an explicit CLI-only request
- for train, `batch` was not asked and was silently defaulted
- for train, `imgsz` was not asked and was silently defaulted
- for train, `model` size was not asked and was silently defaulted
- for train, user provided a raw data directory (not a YAML) and training was started without first completing dataset-readiness checks
- for train, user's images are long horizontal strips but sliding-window crop was never mentioned
- for train, proceeding without showing the pre-execution parameter summary and getting explicit confirmation
- for predict or export, `imgsz` is guessed without first checking the sibling `args.yaml` when that path should exist
- changing `imgsz` after tensor or shape mismatch style errors without asking the user
- using mixed bucket names such as `runs/detect_predict`, `runs/detect_val`, or `runs/detect_export`
- using generic default output paths such as `runs/train*`, `runs/predict*`, `runs/val*`, or `runs/export*` instead of the company path formula
- adding unsupported export arguments such as generic `project=` or `name=` to `yolo export ...`
