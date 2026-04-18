# Val Mode

Official source: `https://docs.ultralytics.com/modes/val/`
Compatibility baseline: `ultralytics v8.4.38`

Use this file when the user wants metrics, plots, checkpoint comparison, or an evaluation pass.

For company path, naming, bucket, and questioning rules, follow [company-cli-standard.md](company-cli-standard.md). If anything in this file appears to conflict with company conventions, the company standard wins.

## What val mode is for

The official docs describe val mode as post-training evaluation used to measure accuracy and generalization, with metrics such as:

- mAP
- precision
- recall
- speed-related outputs

The docs also note that YOLO models remember training settings, so `yolo val model=...` may be enough for models that retain their original dataset configuration.

## Minimum inputs

- `model=...`
- `data=...` when dataset is not already remembered or when the user wants a specific split/dataset

Typical CLI:

```bash
yolo TASK val model=best.pt data=data.yaml imgsz=640
```

Minimal remembered-settings case:

```bash
yolo TASK val model=best.pt
```

## Important official arguments

### Core evaluation controls

- `data=None`
  - dataset YAML for validation
- `imgsz=640`
  - input image size
- `batch=16`
  - batch size
- `conf=0.001`
  - lower than predict defaults because validation needs full PR behavior
- `iou=0.7`
  - IoU threshold for NMS
- `max_det=300`
  - max detections per image

### Runtime

- `half=False`
  - FP16 validation
- `device=None`
  - auto-select best available device
- `dnn=False`
  - OpenCV DNN inference for ONNX
- `workers=8`
- `compile=False`

### Analysis and outputs

- `plots=True`
  - generates plots, confusion matrices, PR curves
- `save_json=False`
  - useful for COCO-style evaluation or downstream tooling
- `save_txt=False`
- `save_conf=False`
- `visualize=False`

### Dataset filtering and protocol choices

- `classes=None`
  - evaluate only selected classes
- `rect=True`
  - rectangular batching
- `split='val'`
  - can be `val`, `test`, or `train`
- `augment=False`
  - test-time augmentation
- `agnostic_nms=False`
- `single_cls=False`
- `end2end=None`

### Output organization

- `project`
- `name`
- `verbose=True`

### Additional official validation arguments

- `workers=8`
- `compile=False`
- `end2end=None`

## Recommended decision rules

- User asks "模型效果怎么样" or "给我指标":
  - use `val`
- User wants benchmark-style accuracy on held-out data:
  - ensure `data=` and likely `split=test` or `split=val`
- User wants plots and confusion matrix:
  - keep `plots=True`
- User wants machine-readable evaluation:
  - add `save_json=True`

## Expected outputs

Usually validation metrics plus:

- confusion matrix
- PR-related plots
- optional JSON or TXT outputs

## What to say in the final response

Include:

1. inferred reason for using `val`
2. exact CLI
3. the main metrics or plots they should expect
4. a next step such as threshold tuning, predict, or export
