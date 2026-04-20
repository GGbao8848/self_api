---
name: sop-workflow
description: Use this skill when the user needs a combined end-to-end workflow across both dataset processing and model CLI work in this self_api project, or when the request is vague enough that you must collect parameters, decide whether data preparation is still needed, then route stages to data-preprocess and ultralytics-yolo-modes. Always present the full stage plan (what runs, parameters, outputs) and wait for user confirmation before any API execution — never run preprocess or training in the first reply. Do not use this skill for pure data-only requests or pure model-CLI-only requests.
---

# Self API Workflow SOP

## Core Design Principle

**Users do not need to memorize project conventions.**

Before routing to any downstream skill or executing any stage — regardless of how much information the user has already provided — this skill MUST always present a complete stage-by-stage execution plan and ask for confirmation.

The plan must cover:

1. **Which stages will run**: list each stage in order (e.g., dataset split → sliding-window crop → XML conversion → training)
2. **What each stage will do**: a plain-language description — no jargon, no convention references the user needs to look up
3. **What parameters each stage will use**: with plain-language explanation of what each field controls and why it matters for the user's data
4. **What results each stage will produce**: exact output paths and artifact names
5. **What the user can change**: list any fields the user may want to adjust before the plan runs

Then ask: "以上流程和参数是否符合你的预期？需要调整哪些步骤或参数？" (Does this plan match your expectations? Which steps or parameters, if any, do you want to change?)

Only after the user confirms — or explicitly says "直接跑" / "确认" / "go ahead" — should this skill route to downstream skills or execute.

### Hard gate — no side effects before the plan

The following are **forbidden before** the user has seen the full multi-stage plan and explicitly confirmed:

- Any HTTP call to `self_api` preprocess endpoints (`xml-to-yolo`, `split-yolo-dataset`, `build-yolo-yaml`, `yolo-train`, async variants, etc.)
- Shell commands such as `curl` / `wget` that hit those endpoints
- Starting training or long-running jobs

**User phrases do not skip the plan.** Even if the user says `用 skill 跑通`, `本地训练`, `从原始数据到训练`, the **first** assistant message must still be the plan (or missing-field questions + plan outline), not execution.

**Turn order**: plan + field meanings + ask for confirmation → user confirms → then execute stage by stage.

---

Use this skill as the combined-workflow orchestrator for this project.

This skill is for:

- receiving a cross-stage or still-ambiguous high-level goal
- collecting missing parameters and path information
- choosing the right standard SOP
- routing the task to the correct downstream skill
- documenting where preprocessing stops and model execution begins

This skill does not own the low-level implementation details.

Do not use this skill when the request is already clearly one-stage:

- data-only requests should go directly to `$data-preprocess`
- model-CLI-only requests should go directly to `$ultralytics-yolo-modes`

After routing:

- use `$data-preprocess` for dataset preprocessing, dataset consolidation, transfer, packaging, and self_api endpoint selection
- use `$ultralytics-yolo-modes` for training-stage parameter normalization and execution-shape selection

## Routing order

- Read [references/project-structure-kb.md](references/project-structure-kb.md) first for stable directory and naming rules.
- Read [references/high-frequency-sops.md](references/high-frequency-sops.md) first.
- Read [references/parameter-checklist.md](references/parameter-checklist.md) before deciding what information is still missing.
- Read [references/workflow-map.md](references/workflow-map.md) whenever the user mentions orchestration, LangGraph pipeline, HITL, SOP templates, or asks how SOPs map to endpoints. (n8n is no longer used by this project.)

## Working rules

- Start by classifying the request into one of three buckets: data preprocessing, model CLI generation, or combined workflow.
- If the request is clearly data-only, stop routing here and switch directly to `$data-preprocess`.
- If the request is clearly model-CLI-only, stop routing here and switch directly to `$ultralytics-yolo-modes`.
- Collect all missing parameters in ONE message, not drip-fed. Batch every missing-field question into a single structured request.
- Default to the smallest complete SOP that fits the goal.
- Keep the answer sequence-first: first determine the stage, then the SOP, then build the full plan.
- When the user is deciding between SOPs, compare by input layout, whether labels are VOC XML or YOLO TXT, whether the task is single-class, whether remote transfer is required, and whether the user is already at the training stage.
- If the dataset is not ready, route to `$data-preprocess`.
- If the dataset is ready and the user needs model commands, route to `$ultralytics-yolo-modes`.
- If the task spans both stages, first route through `$data-preprocess`, then hand off to `$ultralytics-yolo-modes`.
- Always present the full multi-stage plan in one message before executing any stage.
- Before the training stage, decide whether execution is local or remote.
- If training is local, the final injected API target is `POST /api/v1/preprocess/yolo-train/async`（轮询 `/api/v1/tasks/{task_id}`）.
- If training is remote, the final injected API target is `POST /api/v1/preprocess/remote-sbatch-yolo-train`.
- For end-to-end orchestration with HITL, prefer the LangGraph pipeline: `POST /api/v1/pipeline/run`.
- Do not leave this decision implicit. Training-stage requests must state or collect whether execution is local or remote.

## Output style

All responses follow the **plan → explain → confirm → route** sequence.

**Never skip the plan step, even if the user seems to have provided all parameters.**

### For vague or underspecified requests

First collect the missing parameters in one message (not drip-fed). Once enough information is available, build and present the full stage plan.

### For operational requests

Return the chosen SOP as a numbered stage plan in plain language:

```
阶段 1 — 数据集分割
  操作：将原始图片按 80/20 比例分为训练集和验证集
  输入：/path/to/raw/images
  输出：<root_dir>/<detector_name>/datasets/<version>/images/train, .../val
  可调整：train/val 比例（默认 0.8/0.2）

阶段 2 — 滑窗裁剪（如适用）
  操作：对横向长条图进行滑动窗口裁剪，生成固定尺寸小图
  参数：tile_size=640×640, overlap=0.2
  输出：<root_dir>/<detector_name>/datasets/<version>_tiles/
  可调整：tile_size、overlap 比例

阶段 3 — 训练
  操作：调用 POST /api/v1/preprocess/yolo-train（本地）
  参数：model=yolo11s.pt, batch=16, imgsz=640, epochs=100
  输出：<root_dir>/<detector_name>/runs/detect/<name>/weights/best.pt
  可调整：model 规模、batch、imgsz、epochs
```

End with: "以上流程和参数是否符合你的预期？需要调整哪些步骤或参数？确认后我将开始执行。"

### For implementation requests

After confirmation, hand off each stage to the appropriate downstream skill. Do not expand every low-level command here.

### For workflow refactors

Preserve the project's current standard chains unless the user explicitly wants a new convention.

## Trigger examples

Use this skill for requests like:

- `帮我从原始数据一路整理到训练，给我完整 SOP`
- `这批数据还没处理，但最后我要训练，帮我把整个流程串起来`
- `我不确定现在该先清洗数据还是直接训练，你先帮我判断`
- `给我一个长 SOP，包含数据整理、传输、训练命令生成`

Do not use this skill for requests like:

- `只帮我处理数据集`
- `数据已经好了，直接给我 yolo train 命令`
