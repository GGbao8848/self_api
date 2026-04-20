---
name: ultralytics-yolo-modes
description: "Use this skill when the user wants to train, validate, predict, or export an Ultralytics YOLO model and the task needs normalized training parameters, naming, project paths, and execution shape. Always show a full execution preview (operations, parameters, outputs, field meanings) and wait for user confirmation before calling any API or running training — never execute preprocess or yolo-train in the first reply. For train requests in this self_api project, first decide whether execution is local or remote: local training injects the `yolo-train` API, remote training injects the `remote-sbatch-yolo-train` API. Direct raw CLI output is mainly for val, predict, export, or when the user explicitly asks for CLI only."
---

# Ultralytics YOLO Modes

## Core Design Principle

**Users do not need to memorize project conventions.**

Before executing anything — regardless of how much information the user has already provided — the skill MUST always present a complete execution preview and ask for confirmation.

The preview must cover:

1. **What operations will run**: which API endpoint or CLI command, in what order
2. **What parameters will be used**: list every field with its resolved value and a plain-language explanation of what it does and why it matters
3. **What results will be produced**: exact output paths, artifact names (e.g., `best.pt`), and what each artifact is for
4. **What the next recommended step is**: e.g., validate, export, deploy

Then ask: "以上参数和操作是否符合你的预期？需要调整哪些内容？" (Do these parameters and operations match your expectations? Which fields, if any, do you want to change?)

Only after the user confirms — or explicitly says "直接跑" / "确认" / "go ahead" — should the skill execute or output the final payload/CLI.

This pattern applies to ALL modes: `train`, `val`, `predict`, and `export`.

### Hard gate — no side effects before preview

The following count as **execution** and are **forbidden in the first assistant turn** (and forbidden before the user has replied to the preview with explicit confirmation):

- Calling `POST` / `GET` to any `self_api` preprocess endpoint (including `xml-to-yolo`, `split-yolo-dataset`, `publish-yolo-dataset`, `yolo-train`, `yolo-train/async`, etc.)
- Running shell commands that invoke `curl`, `wget`, `httpie`, or Python `requests` against the local API
- Running `conda run ... yolo train`, raw `yolo train`, or any subprocess that starts training
- Editing or writing files under dataset or `runs/` directories when the intent is to **perform** the pipeline instead of only describing it

**Phrases from the user do NOT waive this gate.** Examples that still require a full preview first, then a **second** turn after user confirmation:

- `用 skill 训练`, `按 skill 来`, `本地训练`, `虚拟环境 yolo_pose`
- `帮我跑完`, `开始吧`, `直接训练` (these are **not** confirmation of parameters until a preview table has been shown in **this** conversation turn)

**Correct turn order**

1. **First reply**: either (a) one message that lists missing fields you still need, or (b) the **full execution preview** (operations + parameters + paths + field meanings + ask for confirmation). No API calls, no training.
2. **After the user confirms** (or adjusts parameters and confirms again): then output payload / CLI, or run commands / call API as agreed.

If the user has not yet seen a preview of **what will run** and **what will be produced**, you are not allowed to execute — even if you already inspected files on disk.

---

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

The following fields must be resolved before any preview can be shown.

Fields that can be inferred from local files or context (infer first, then show value in preview for user to confirm):

- `task`: infer from dataset labels (`detect`, `segment`, `classify`, `pose`, or `obb`)
- `project` and `name`: derived from `root_dir + detector_name + bucket` rules; never ask the user to spell these out
- `imgsz` for `val`/`predict`/`export`: read from sibling `args.yaml` first; fall back to `640`

Fields that MUST be asked if not explicitly provided by the user (do NOT silently infer or default):

- `mode`: `train`, `val`, `predict`, or `export`
- `root_dir`: business project root directory — ask if not given
- `detector_name`: stable model family name — ask if not given
- `model`: for train, ask size preference (nano/small/medium/large/x) or custom `.pt` path
- `batch`: for train, always ask — never silently default
- `imgsz`: for train, always ask — remind that strip/panoramic images benefit from non-square sizes
- `epochs`: for train, ask or confirm a speed-vs-quality preference
- execution target: local or remote — ask if ambiguous
- data input: dataset YAML path for train/val; image/folder/stream path for predict; model path for export

If the task type cannot be reliably inferred, ask one focused question rather than guessing.

## Workflow

**Step 0 — Pre-flight intake interview (MANDATORY for `train`)**

Before doing anything else for a `train` request, run a structured intake interview.
Do NOT skip this step even if the user has provided some information.
Do NOT produce any API payload, CLI, or execution until ALL mandatory fields are confirmed.

Procedure:

1. Check whether the user has provided a ready dataset YAML or only a raw data path (image folder, xml folder, or unlabeled directory).
   - If only a raw path is given, STOP. Route to `$data-preprocess` or `$sop-workflow` first.
   - Raw path signals: ends in `/images`, `/xmls`, `/annotations`, or contains image files like `.jpg`/`.png` directly.
   - Dataset YAML signal: path ends in `.yaml` or `.yml` and contains `train:`, `val:`, `nc:`.
2. If raw data is given, ask these data-readiness questions first (collect all in one message, don't drip-feed):
   - Has the dataset already been split into train/val sets?
   - Are the images long horizontal strips or panoramic scenes? If yes, recommend sliding-window crop and ask whether to enable it.
   - Does the dataset need XML-to-YOLO conversion?
   - Where should the processed dataset be stored?
3. Once the dataset YAML is confirmed or will be produced by preprocessing, collect all training fields in one message:
   - `detector_name`: the stable model family name used for directory structure
   - `root_dir`: the business project root directory where runs will be saved
   - `model`: pretrained checkpoint size preference (nano/small/medium/large/x, or custom `.pt` path)
   - `batch`: training batch size (ask if not given; do not default silently)
   - `imgsz`: training image size (ask if not given; remind that strip images often use non-square sizes like 1280×320)
   - `epochs`: total training epochs or a speed-vs-quality preference
   - execution target: local or remote
4. Before executing, show the user a summary table of all collected parameters and explain what each will produce (e.g., "batch=16 + imgsz=640 will fit on a single 8 GB GPU and train for approximately X hours"). Ask for explicit confirmation.
5. Only after confirmation, generate the final API payload or CLI.

**Step 1 — Mode identification**

Identify whether the user wants model creation, evaluation, inference, or deployment.

**Step 2 — Execution target**

Resolve local or remote execution first.

**Step 3 — Task and mode selection**

Pick the YOLO task and mode.

**Step 4 — Reference loading**

Load [../sop-workflow/references/project-structure-kb.md](../sop-workflow/references/project-structure-kb.md) for shared project structure, then [references/company-cli-standard.md](references/company-cli-standard.md) for path and naming normalization.

**Step 5 — API injection target (train only)**

For `train`, decide the API injection target:
- local training -> `yolo-train`
- remote training -> `remote-sbatch-yolo-train`

**Step 6 — Mode reference**

Load the matching mode reference and add only the arguments that materially affect the user's goal.

**Step 7 — Export constraint**

For `export`, follow the official CLI argument surface from the export docs. Do not invent generic `project=` or `name=` arguments, because export mode does not expose them as normal documented CLI parameters.

**Step 8 — Explain before executing**

Tell the user what the final API payload or CLI will produce, what artifacts will appear and where, and what the recommended next step is (e.g., validate, export). For `train`, always include the expected output directory and key artifact path.

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

Every response follows the **preview → explain → confirm → execute** sequence.

**Never skip the preview step, even if the user has already provided all parameters.**

### Step A — Preview

State in plain language:

- inferred task, mode, and execution target
- the full parameter table (see format below)
- expected output directory, key artifact path, and what the artifact is for
- recommended next step after this operation

### Step B — Parameter table format

For every field, show:

| 字段 | 值 | 含义 |
|---|---|---|
| `model` | `yolo11s.pt` | Small 规模预训练权重，精度与速度均衡，适合小目标检测 |
| `batch` | `16` | 每次送入 GPU 的图片数，影响显存占用和梯度质量 |
| `imgsz` | `640` | 训练分辨率，越大越慢但对小目标更友好 |
| `epochs` | `100` | 总训练轮数，决定模型学习的充分程度 |
| `project` | `/data/TVDS/detector/runs/detect` | 运行结果的父目录，由 root_dir + detector_name + bucket 推导 |
| `name` | `detector_20260418_1430` | 本次运行的子目录名，与数据集 YAML 文件名对齐 |

For `train`, also show the derived output path:

```
训练结果将保存至：<project>/<name>/weights/best.pt
```

### Step C — Confirmation

End every preview with:

> 以上参数和操作是否符合你的预期？需要调整哪些内容？确认后我将执行。

### Step D — Execute

Only after the user confirms, generate and execute the final API payload or CLI.

Accepted confirmation signals: "确认", "没问题", "直接跑", "go ahead", "ok", "好的", or equivalent.

If the user requests changes, update the affected fields, regenerate the preview, and ask for confirmation again.

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
