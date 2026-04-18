# Export Mode

Official source: `https://docs.ultralytics.com/modes/export/`
Compatibility baseline: `ultralytics v8.4.38`

Use this file when the user needs a deployment artifact from a trained or pretrained YOLO model.

For company path, naming, bucket, and questioning rules, follow [company-cli-standard.md](company-cli-standard.md). If anything in this file appears to conflict with company conventions, the company standard wins.

Company default for final CLI generation:

- `format=torchscript`
- `half=True`

## What export mode is for

The official docs position export mode as the bridge from PyTorch checkpoints to deployment formats. The page explicitly highlights:

- ONNX and OpenVINO for CPU-oriented speedups
- TensorRT for GPU-oriented speedups
- CoreML and mobile-oriented targets
- quantization and dynamic shape options

## Minimum inputs

- `model=...`
- `format=...`

Canonical CLI:

```bash
yolo export model=best.pt format=torchscript half=True
```

Important constraint from the official docs:

- the general export CLI surface is `yolo export model=... format=... [export args]`
- do not add generic `project=` or `name=` arguments to export commands
- one documented exception in the export format table is `format=rknn`, which supports a `name` argument

## Important official arguments

### Core export controls

- `format='torchscript'`
  - target format such as `onnx`, `torchscript`, `engine`, `openvino`, `coreml`, `tflite`
- `imgsz=640`
  - input size
- `batch=1`
  - batch inference size supported by exported model

### Optimization and quantization

- `half=False`
  - FP16 quantization
- `int8=False`
  - INT8 quantization
- `optimize=False`
  - mobile-oriented TorchScript optimization
- `simplify=True`
  - ONNX graph simplification
- `dynamic=False`
  - dynamic input shapes
- `opset=None`
  - ONNX opset version
- `nms=False`
  - embed NMS when supported
- `end2end=None`

### Hardware and backend-specific knobs

- `device=None`
  - `0`, `cpu`, `mps`, `npu`, `dla:0`, `dla:1`
- `workspace=None`
  - TensorRT workspace in GiB
- `keras=False`
  - SavedModel/Keras-related export path

### Calibration inputs for INT8

- `data='coco8.yaml'`
  - used for INT8 calibration if needed
- `fraction=1.0`
  - dataset fraction for INT8 calibration

### General behavior reminders

- exported artifacts are typically written next to the source checkpoint or into a format-specific sibling directory
- describe the expected artifact path from the model path and format rather than fabricating a separate export run bucket

## Important official format table

The official export page lists these notable formats and artifact styles:

- `onnx` -> `model.onnx`
- `engine` -> `model.engine`
- `openvino` -> `model_openvino_model/`
- `coreml` -> `model.mlpackage`
- `saved_model` -> `model_saved_model/`
- `tflite` -> `model.tflite`
- `tfjs` -> `model_web_model/`
- `ncnn` -> `model_ncnn_model/`
- `executorch` -> `model_executorch_model/`

## Recommended decision rules

- User says "部署到通用推理引擎 / 跨框架":
  - company default is still `format=torchscript half=True` unless the user explicitly asks for another backend
- User says "NVIDIA GPU 线上部署":
  - consider `format=engine`
- User says "CPU 边缘设备":
  - consider `format=openvino` or `format=onnx`
- User says "Apple 生态":
  - consider `format=coreml`
- User says "量化/更小模型":
  - consider `half=True` or `int8=True`
- User says "输入尺寸不固定":
  - consider `dynamic=True`

## Format-specific reminders from official docs

- ONNX commonly pairs with `simplify=True`
- TensorRT supports `workspace`, `int8`, `dynamic`, and auto-uses GPU
- INT8 export may require `data=` for calibration; official fallback is `coco8.yaml`
- `dynamic=True` is automatically enabled when TensorRT export uses INT8

## Typical commands

ONNX:

```bash
yolo export model=best.pt format=onnx imgsz=640 dynamic=True simplify=True
```

TensorRT:

```bash
yolo export model=best.pt format=engine imgsz=640 half=True device=0
```

OpenVINO INT8:

```bash
yolo export model=best.pt format=openvino int8=True data=data.yaml
```

## Expected outputs

Depends on format, usually next to the source checkpoint or in format-specific directories:

- `.onnx`
- `.engine`
- `_openvino_model/`
- `.mlpackage`
- `.tflite`

For company-standard CLI generation:

- default export format is `torchscript`
- default export precision is `half=True`
- do not inject `project` or `name` unless the chosen export format explicitly documents that argument

## What to say in the final response

Include:

1. target deployment environment and why this format matches it
2. exact CLI
3. exported artifact name or directory pattern
4. how to run a quick sanity check on the exported file
