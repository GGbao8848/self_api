---
name: data-preprocess
description: Use this skill when the user wants to process datasets with this self_api project, especially dataset preprocessing, nested-dataset cleanup, XML-to-YOLO conversion, dataset aggregation, visualization, split, augmentation, sliding-window crop, YAML generation, zip/unzip, move/copy, or cross-server SFTP transfer. Prefer the project's synchronous preprocess API commands unless the user explicitly asks for async jobs, callbacks, or n8n orchestration.
---

# Self API Data Preprocess

Use this skill for the data side of this API project.

This skill is for:

- dataset preprocessing and cleanup
- nested dataset consolidation
- XML to YOLO conversion
- YOLO dataset checking, split, augmentation, and sliding-window crop
- dataset YAML generation
- zip/unzip, move/copy, and cross-server transfer

This skill is not for model CLI generation. If the user needs `yolo train`, `yolo val`, `yolo predict`, or `yolo export` commands, switch to `$ultralytics-yolo-modes`.

## Default behavior

- Prefer synchronous `curl` commands.
- Only use `/async` endpoints when the user explicitly asks for callbacks, polling, automation, or long-running workflow execution.
- Keep answers operational: map the user's goal and local paths to the shortest valid API call sequence.
- Reuse the standard workflow chains in [references/common-chains.md](references/common-chains.md).

## What to read

- Read [../sop-workflow/references/project-structure-kb.md](../sop-workflow/references/project-structure-kb.md) when output naming, dataset YAML naming, transfer target layout, or project directory conventions matter.
- Read [references/common-chains.md](references/common-chains.md) first to choose the right preprocessing chain.
- Read [references/sync-api-commands.md](references/sync-api-commands.md) for exact synchronous `curl` examples.
- Read [references/request-notes.md](references/request-notes.md) when you need field constraints or parameter rules.

## Working rules

- Assume the project API base URL must be explicit in the final command. If the user does not give one, keep the example base URL or ask for it only when execution is required.
- Preserve the project's standard directory assumptions such as `images/`, `xmls/`, `labels/`, and optional `backgrounds/`.
- When the user does not specify a special location, prefer dataset outputs and versioned dataset folders under `<root_dir>/<detector_name>/datasets`.
- When a versioned dataset folder is created, make `<dataset_version>` equal the dataset YAML filename stem in that folder.
- When the request is underspecified, ask only for the missing paths or the dataset layout choice that changes the endpoint parameters materially.
- For cross-server delivery, prefer `zip-folder -> remote-transfer -> remote-unzip` unless the user clearly wants raw directory transfer.
- For single-class YOLO cleanup, use `reset-yolo-label-index` before split or crop when that matches the workflow.

## Trigger examples

Use this skill for requests like:

- `帮我把这批多层目录图片和 xml 整理成标准数据集`
- `把 /path/raw_data 清洗成 images/xmls，再转成 yolo`
- `我只想做数据预处理，不要训练`
- `把这个数据集切分、增强、生成 yaml`
- `帮我把数据打包传到另一台服务器并远程解压`

Do not use this skill for requests like:

- `帮我生成 yolo train 命令`
- `我已经有 dataset.yaml 了，直接给我训练 cli`
