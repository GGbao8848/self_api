# Project Structure KB

This file is the stable project-management knowledge base for directory structure and naming.

Treat it as macro-level convention, not per-task SOP.

Use it when a task depends on:

- project root layout
- detector naming
- dataset storage layout
- run-directory naming
- version naming
- output bucket selection

## Core fields

- `root_dir`: business project root directory
- `detector_name`: stable detector or model family name
- `name`: current version name or run name

## Naming semantics

Do not mix these fields:

- `detector_name`
  - stable detector family
  - used in the project path
- `name`
  - current dataset version or run version
  - used as the leaf run directory

For training, the preferred convention is:

- dataset YAML filename should usually be `detector_name + YYYYMMDD_HHMM`
- `name` should equal the dataset YAML stem

Example:

- `detector_name=nzxj_louyou`
- dataset YAML: `nzxj_louyou_20260417_1430.yaml`
- final `name=nzxj_louyou_20260417_1430`

## Run bucket dictionary

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

Final run directory:

```text
<project>/<name>
```

Example:

```text
/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect/nzxj_louyou_20260417_1430
```

## Dataset storage rule

Datasets should usually live under:

```text
<root_dir>/<detector_name>/datasets
```

Use this directory as the standard dataset-version management area.

Preferred pattern:

```text
<root_dir>/<detector_name>/datasets/<dataset_version>
```

Hard rule:

- `<dataset_version>` must equal the dataset YAML filename stem inside that folder
- do not use a dataset folder name that differs from its `dataset.yaml`-style version filename stem

Common examples:

```text
<root_dir>/<detector_name>/datasets/nzxj_louyou_20260417_1430
<root_dir>/<detector_name>/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml
<root_dir>/<detector_name>/datasets/nzxj_louyou_20260417_1430/images
<root_dir>/<detector_name>/datasets/nzxj_louyou_20260417_1430/labels
```

If a task needs dataset YAML naming, transfer target layout, or dataset output placement, prefer this dataset area unless the user explicitly asks for another location.

That means the preferred alignment is:

- dataset folder: `nzxj_louyou_20260417_1430`
- dataset YAML: `nzxj_louyou_20260417_1430.yaml`
- dataset version: `nzxj_louyou_20260417_1430`

## Stability rule

These rules are knowledge-base level and should change rarely.

Workflow order may change by task, but this directory and naming convention should remain the shared upstream rule unless the project itself changes.

## Usage by skill

- `sop-workflow`: use this file to understand stable project structure before choosing a long SOP
- `data-preprocess`: use this file when preprocessing outputs, dataset YAML naming, or transfer targets should follow project convention
- `ultralytics-yolo-modes`: use this file as the upstream directory and naming authority before applying mode-specific CLI rules
