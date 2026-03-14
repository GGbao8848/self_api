---
name: teds-n8n-nested-dataset-workflow
description: Build, update, and troubleshoot the TEDS nested dataset n8n workflow stored in docs/n8n_nested_dataset_workflow.json. Use when editing this workflow, parameterizing input_dir or api_base_url, adapting it for Docker-to-host networking, converting synchronous steps to async/callback orchestration, or fixing n8n expression and item-pairing issues in HTTP Request and Code nodes.
---

# TEDS N8N Nested Dataset Workflow

Use this skill when working on the repository workflow JSON at `docs/n8n_nested_dataset_workflow.json`.

## Default Goal

Maintain a working n8n pipeline for this processing order:

1. Discover leaf directories
2. Clean nested dataset
3. Convert cleaned fragments with `xml-to-yolo`
4. Aggregate into final dataset

## Core Files

- Main workflow: `docs/n8n_nested_dataset_workflow.json`
- API routes: `app/api/v1/endpoints/preprocess.py`
- Request/response models: `app/schemas/preprocess.py`
- Nested dataset services: `app/services/nested_dataset.py`
- API examples: `docs/api_examples.md`

For node-by-node mapping and common fixes, read `references/workflow-map.md`.

## Workflow Editing Rules

1. Treat `docs/n8n_nested_dataset_workflow.json` as the source of truth unless the user explicitly asks for a new variant.
2. Keep `api_base_url` configurable. When n8n runs in Docker and the API runs on the host, use the host IP rather than `127.0.0.1`.
3. Prefer passing shared values forward in item data over repeatedly referencing a single-item config node from a multi-item node.
4. When a node fans out to multiple items, avoid expressions like `{{$node["Set Config"].json...}}` inside downstream multi-item HTTP Request parameters unless item pairing is guaranteed.
5. If a step may run for a long time, prefer one of these patterns:
   - API async endpoint + callback webhook
   - API async endpoint + bounded polling
6. Preserve path semantics: the API executes on the host, so all `input_dir` and `output_dir` values must be host filesystem paths.

## Common Failure Patterns

### Paired item index errors

Typical error:

- `pairedItemInvalidIndex`
- `node has 1 item(s) but you're trying to access item N`

Fix:

1. Find which node fans out into multiple items.
2. Move shared config values such as `api_base_url`, `include_difficult`, `overwrite`, and directory names into each emitted item.
3. Change downstream expressions to read from `$json` instead of `$node["Set Config"]`.

### Docker networking mistakes

If n8n is in Docker and the API is on the host:

- Do not use `http://127.0.0.1:8666`
- Use `http://<host-ip>:8666/api/v1`

### Long request timeout risk

If `clean-nested-dataset`, `xml-to-yolo`, or `aggregate-nested-dataset` may take a long time:

1. Switch to `/async`
2. Add `callback_url`
3. Let n8n continue from the callback

## Update Procedure

When modifying the workflow:

1. Read the current workflow JSON.
2. Confirm which steps are sync, async, or callback-driven.
3. Match every API call to an existing endpoint in `app/api/v1/endpoints/preprocess.py`.
4. If parameters are added, confirm them in `app/schemas/preprocess.py`.
5. Keep the workflow importable JSON.
6. Validate JSON syntax after edits.

## Expected Input Shapes

### Fixed-config manual workflow

Use a Code node that returns one item containing:

```json
{
  "api_base_url": "http://192.168.2.26:8666/api/v1",
  "input_dir": "/media/qzq/16T/...",
  "clean_output_dir": "/media/qzq/16T/.../cleaned_dataset",
  "aggregated_output_dir": "/media/qzq/16T/.../dataset"
}
```

### Parameterized webhook workflow

Expect request body fields such as:

```json
{
  "api_base_url": "http://192.168.2.26:8666/api/v1",
  "n8n_webhook_base": "http://192.168.2.26:5678",
  "input_dir": "/media/qzq/16T/..."
}
```

## Output Expectations

A finished workflow should surface at least:

- `input_dir`
- `clean_output_dir`
- `dataset_output_dir`
- `aggregated_images`
- `aggregated_backgrounds`
- `classes_file`
- `manifest_path`

## When Creating Variants

If the user asks for a new version, name it by orchestration mode:

- `...workflow.json` for the base pipeline
- `...workflow-async.json` for polling-based async
- `...workflow-callback.json` for callback-driven orchestration

Keep the same processing semantics unless the user asks for behavior changes.
