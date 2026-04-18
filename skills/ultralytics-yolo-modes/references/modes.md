# Ultralytics YOLO Modes Router

Use this file only to choose the correct mode quickly, then immediately load the detailed reference for that mode.

Source basis: official Ultralytics modes overview and mode-specific pages:

- `https://docs.ultralytics.com/modes/`
- `https://docs.ultralytics.com/modes/train/`
- `https://docs.ultralytics.com/modes/val/`
- `https://docs.ultralytics.com/modes/predict/`
- `https://docs.ultralytics.com/modes/export/`

## Route the request

- User wants to build a custom model from labeled data -> `train`
- User wants metrics, mAP, PR curves, confusion matrix, or checkpoint comparison -> `val`
- User wants inference on images, folders, video, webcam, or stream -> `predict`
- User wants ONNX, TensorRT, OpenVINO, CoreML, TFLite, or other deployable artifact -> `export`

## Then load one detailed reference

- `train` -> [train.md](train.md)
- `val` -> [val.md](val.md)
- `predict` -> [predict.md](predict.md)
- `export` -> [export.md](export.md)

## Fast reminders from the official overview

- Train fine-tunes on custom or preloaded datasets
- Val evaluates generalization and reports metrics such as mAP
- Predict runs inference on new media and streams
- Export converts PyTorch checkpoints into deployment formats

## Minimal command skeletons

```bash
yolo TASK train model=MODEL data=DATA_YAML
yolo TASK val model=MODEL data=DATA_YAML
yolo TASK predict model=MODEL source=SOURCE
yolo export model=MODEL format=FORMAT
```
