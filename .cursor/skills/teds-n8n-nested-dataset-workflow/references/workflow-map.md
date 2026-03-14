# Workflow Map

Reference target: `docs/n8n_nested_dataset_workflow.json`

## Main Node Chain

1. `Manual Trigger`
2. `Set Config`
3. `Discover Leaf Dirs`
4. `Validate Leaf Dirs`
5. `Clean Nested Dataset`
6. `Build XML To YOLO Inputs`
7. `XML To YOLO`
8. `Summarize Fragment Conversion`
9. `Aggregate Nested Dataset`
10. `Final Summary`

## API Mapping

### `Discover Leaf Dirs`

- Endpoint: `POST /api/v1/preprocess/discover-leaf-dirs`
- Purpose: find the deepest directories that directly contain image or XML files

### `Clean Nested Dataset`

- Endpoint: `POST /api/v1/preprocess/clean-nested-dataset`
- Purpose: split nested raw data into `images`, `xmls`, and `backgrounds`

### `XML To YOLO`

- Endpoint: `POST /api/v1/preprocess/xml-to-yolo`
- Purpose: generate `labels` from cleaned fragment directories

### `Aggregate Nested Dataset`

- Endpoint: `POST /api/v1/preprocess/aggregate-nested-dataset`
- Purpose: merge all fragment datasets into one final dataset

## Shared Config Fields

The workflow commonly carries these values:

- `api_base_url`
- `input_dir`
- `clean_output_dir`
- `aggregated_output_dir`
- `images_dir_name`
- `xmls_dir_name`
- `labels_dir_name`
- `backgrounds_dir_name`
- `include_difficult`
- `copy_files`
- `overwrite`

## Safe Expression Pattern

For single-item config:

```text
{{$node["Set Config"].json.api_base_url}}
```

For multi-item downstream nodes, prefer:

```text
{{$json.api_base_url}}
```

after copying config values into each emitted item.

## Docker-to-Host Rule

If n8n runs in Docker and the API runs on the host:

- Good: `http://192.168.2.26:8666/api/v1`
- Bad: `http://127.0.0.1:8666/api/v1`

## Typical User Requests This Skill Should Handle

- “把当前 workflow 改成可参数化输入目录”
- “把同步版改成 async 或 callback 版”
- “修复 xml-to-yolo 节点的表达式报错”
- “把 API 地址改成宿主机 IP”
- “新增一个通知节点”
- “把当前工作流拆成开始流和回调流”
