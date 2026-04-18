# n8n Workflow Map

These project files are the closest workflow artifacts for orchestration-oriented tasks:

- `docs/n8n/self_api_小图(images_xmls)_baseline模型训练.json`
- `docs/n8n/self_api_小图本地训练.json`
- `docs/n8n/self_api_大图(images_xmls)_常发模型迭代.json`
- `docs/n8n/self_api_大图滑窗裁剪本地训练.json`

Use them when the user asks for:

- n8n workflow mapping
- node-by-node endpoint order
- callback-oriented async orchestration
- a JSON workflow that matches one of the standard SOPs

## Mapping from SOP to n8n artifacts

### Small-image baseline

- Prefer `self_api_小图(images_xmls)_baseline模型训练.json`
- Local-training variant: `self_api_小图本地训练.json`

### Large-image iteration

- Prefer `self_api_大图(images_xmls)_常发模型迭代.json`
- Sliding-window local-training variant: `self_api_大图滑窗裁剪本地训练.json`

## How to use this reference

- Start from the SOP in [high-frequency-sops.md](high-frequency-sops.md).
- Then inspect the closest JSON file only if the user needs n8n-specific structure or async details.
- Do not load all workflow JSON files unless the task is explicitly about comparing them.
