---
name: sop-workflow
description: Use this skill when the user needs a combined end-to-end workflow across both dataset processing and model CLI work in this self_api project, or when the request is vague enough that you must collect parameters, decide whether data preparation is still needed, then route stages to data-preprocess and ultralytics-yolo-modes. Do not use this skill for pure data-only requests or pure model-CLI-only requests.
---

# Self API Workflow SOP

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
- Read [references/n8n-workflow-map.md](references/n8n-workflow-map.md) if the user mentions n8n, orchestration, or wants the JSON workflow equivalents.

## Working rules

- Start by classifying the request into one of three buckets: data preprocessing, model CLI generation, or combined workflow.
- If the request is clearly data-only, stop routing here and switch directly to `$data-preprocess`.
- If the request is clearly model-CLI-only, stop routing here and switch directly to `$ultralytics-yolo-modes`.
- Collect only the parameters that are necessary to continue. Prefer short clarification questions over broad interviews.
- Default to the smallest complete SOP that fits the goal.
- Keep the answer sequence-first: first determine the stage, then the SOP, then the downstream skill.
- When the user is deciding between SOPs, compare by input layout, whether labels are VOC XML or YOLO TXT, whether the task is single-class, whether remote transfer is required, and whether the user is already at the training stage.
- If the dataset is not ready, route to `$data-preprocess`.
- If the dataset is ready and the user needs model commands, route to `$ultralytics-yolo-modes`.
- If the task spans both stages, first route through `$data-preprocess`, then hand off to `$ultralytics-yolo-modes`.
- Use a long SOP only when the user needs both stages or when the current stage is still unclear.
- Before the training stage, decide whether execution is local or remote.
- If training is local, the final injected API target is `POST /api/v1/preprocess/yolo-train`.
- If training is remote, the final injected API target is `POST /api/v1/preprocess/remote-sbatch-yolo-train`.
- Do not leave this decision implicit. Training-stage requests must state or collect whether execution is local or remote.

## Output style

- For vague requests, first list the missing parameters that block routing.
- For operational requests, return the chosen SOP as a short numbered sequence and state which downstream skill should execute each stage.
- For implementation requests, do not expand every low-level command in this skill unless that is required to unblock the route. Hand off to the downstream skill instead.
- For workflow refactors, preserve the project's current standard chains unless the user explicitly wants a new convention.

## Trigger examples

Use this skill for requests like:

- `帮我从原始数据一路整理到训练，给我完整 SOP`
- `这批数据还没处理，但最后我要训练，帮我把整个流程串起来`
- `我不确定现在该先清洗数据还是直接训练，你先帮我判断`
- `给我一个长 SOP，包含数据整理、传输、训练命令生成`

Do not use this skill for requests like:

- `只帮我处理数据集`
- `数据已经好了，直接给我 yolo train 命令`
