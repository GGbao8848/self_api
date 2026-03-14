---
name: api-docs-to-n8n-workflow
description: Design, generate, and troubleshoot n8n workflows from API documentation, OpenAPI specs, Swagger docs, endpoint examples, or request/response schemas. Use when the user asks to create an n8n workflow from API docs, convert API endpoints into HTTP Request nodes, choose between sync/async/callback orchestration, parameterize inputs, or fix n8n expression, item-pairing, and Docker networking issues in API-driven workflows.
---

# API Docs To N8N Workflow

Use this skill to turn API documentation into importable n8n workflows.

## What To Read First

Start from the most authoritative source available:

1. OpenAPI / Swagger schema
2. API reference or README
3. Example requests in docs
4. Server-side route and schema code

If multiple sources exist, resolve conflicts in that order unless runtime code clearly differs.

## Default Workflow Design Process

1. Identify the business goal
2. List required API endpoints in execution order
3. Classify each step as:
   - synchronous request
   - async submit + poll
   - async submit + callback
4. Define the data contract between nodes
5. Choose trigger type:
   - `Manual Trigger` for testing
   - `Webhook` for parameterized execution
   - `Schedule Trigger` for recurring jobs
6. Build the minimal working flow first
7. Add error handling, summary output, and optional notifications

## Node Mapping Rules

### Typical API-to-n8n mapping

- Trigger input: `Webhook`, `Manual Trigger`, or `Schedule Trigger`
- API call: `HTTP Request`
- Data shaping: `Set` or `Code`
- Fan-out over multiple items: `Code` or `Split In Batches`
- Final response to caller: `Respond to Webhook`
- Human-readable output summary: `Code`

### Prefer `Set` vs `Code`

- Use `Set` for simple field mapping
- Use `Code` for loops, conditionals, aggregation, or callback coordination

## Orchestration Choice

### Use synchronous flow when

- Each API call is short
- No long-running server task is involved
- The caller can wait for the full chain

### Use async submit + polling when

- The API exposes `/async` plus task status endpoints
- No callback mechanism is available
- Task duration may exceed a normal HTTP timeout

### Use async submit + callback when

- The API accepts `callback_url`
- The task may be long-running
- You want n8n to avoid long polling loops

Prefer callback over polling when both are available.

## Expression Rules

### Safe referencing

When a node outputs multiple items, do not rely on a single-item config node unless item pairing is guaranteed.

Bad pattern:

```text
{{$node["Set Config"].json.api_base_url}}
```

inside a downstream multi-item node.

Preferred pattern:

1. Copy shared values into each emitted item
2. Read them with:

```text
{{$json.api_base_url}}
```

### Webhook input

For webhook-triggered workflows, request payload is usually under:

```text
{{$json.body}}
```

Inside `Code` nodes, use direct JavaScript access:

```javascript
const payload = $input.first().json.body || {};
```

## Docker Networking Rule

If n8n runs in Docker and the API runs on the host:

- Do not use `127.0.0.1` unless the API is inside the same container
- Prefer the host IP such as `http://192.168.x.x:PORT`
- Or use `host.docker.internal` only if that mapping is known to work

## Generic Workflow Shapes

### Shape 1: Manual testing workflow

1. `Manual Trigger`
2. `Code` or `Set Config`
3. one or more `HTTP Request`
4. `Code` summary

### Shape 2: Parameterized API workflow

1. `Webhook`
2. parse request body
3. one or more `HTTP Request`
4. `Respond to Webhook`

### Shape 3: Callback-driven workflow

Use two webhook paths:

1. start webhook
2. callback webhook

Flow:

1. start webhook receives parameters
2. n8n submits async API task with `callback_url`
3. API calls callback webhook when finished
4. callback webhook triggers the next step or finalizes output

## Required Inputs To Derive From Docs

Before generating the workflow, extract:

- base URL format
- authentication style
- endpoint paths and methods
- request body schema
- query parameters
- response schema
- error schema
- async status endpoint if any
- callback fields if any

If docs are incomplete, inspect server route code and schema definitions.

## Output Contract For Generated Workflows

A generated workflow should make it easy to locate:

- input parameters
- API base URL
- auth handling
- request payload mapping
- async/callback logic
- final output summary
- failure path

## Common Failure Patterns

### `pairedItemInvalidIndex`

Cause:

- multi-item node reads data from a single-item upstream node using paired item semantics

Fix:

1. move shared config into each item
2. switch downstream expressions from `$node[...]` to `$json`

### HTTP timeout on long task

Fix:

1. switch to async endpoint
2. use status polling or callback

### Wrong webhook payload path

Fix:

- for webhook nodes, check whether values are under `$json.body`

### Docker can’t reach host API

Fix:

- replace `127.0.0.1` with host IP

## Delivery Rules

When creating a workflow from API docs:

1. keep the first version minimal and runnable
2. prefer importable JSON output
3. make base URL and input parameters configurable
4. document which endpoint each node corresponds to
5. if there is an existing workflow file, update it instead of creating parallel duplicates unless the user asks for a variant

## Variant Naming

If multiple versions are needed, name them by orchestration mode:

- `...workflow.json`
- `...workflow-async.json`
- `...workflow-callback.json`

## Additional Reference

For a reusable checklist and node design prompts, read `references/workflow-design-checklist.md`.
