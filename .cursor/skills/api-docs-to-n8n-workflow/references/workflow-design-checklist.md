# Workflow Design Checklist

Use this checklist when converting API docs into an n8n workflow.

## 1. Read The API Surface

Capture:

- base URL
- authentication method
- endpoint path + method
- request body shape
- response body shape
- async status endpoint
- callback support

## 2. Choose Trigger

- `Manual Trigger` for local testing
- `Webhook` when external systems should pass parameters
- `Schedule Trigger` for periodic execution

## 3. Decide Orchestration

### Sync

Choose when all API calls are fast and deterministic.

### Async + Poll

Choose when:

- API returns task ids
- API exposes status endpoints
- callback is unavailable

### Async + Callback

Choose when:

- API accepts `callback_url`
- task duration can be long
- you want n8n to avoid holding a long-running poll loop

## 4. Build Node Sequence

Minimal sequence:

1. trigger
2. prepare config
3. call endpoint(s)
4. transform result
5. emit summary

For async:

1. trigger
2. submit async task
3. poll or wait for callback
4. continue next step

## 5. Prevent Common n8n Errors

### Multi-item config access

If an upstream node emits one config item and a later node emits many items:

- avoid downstream references to `$node["Config"]...`
- instead copy config into each item and use `$json...`

### Webhook body confusion

In expression fields:

- use `{{$json.body.field}}`

In Code nodes:

```javascript
const payload = $input.first().json.body || {};
```

### Docker networking

If n8n is containerized and API is on the host:

- use host IP
- avoid `127.0.0.1`

## 6. Validate Before Hand-off

Check that:

- JSON is syntactically valid
- every HTTP node points to a real endpoint
- request fields match the documented schema
- async flow has a completion path
- summary output contains the important result fields

## 7. Good Deliverables

Prefer one of these:

- importable n8n workflow JSON
- updated existing workflow file
- workflow JSON plus a short mapping from nodes to endpoints
