# Predict Mode

Official source: `https://docs.ultralytics.com/modes/predict/`

Use this file when the user wants inference on new images, folders, videos, webcams, streams, or URLs.

For company path, naming, bucket, and questioning rules, follow [company-cli-standard.md](company-cli-standard.md). If anything in this file appears to conflict with company conventions, the company standard wins.

## What predict mode is for

The official docs describe predict mode as real-time or batch inference across many source types, including:

- single image
- image folder
- video file
- webcam index like `0`
- URL
- `.txt` source lists
- multiple streaming sources

The docs also note that inference uses `rect=True` minimal padding by default.

## Minimum inputs

- `model=...`
- `source=...`

Canonical CLI:

```bash
yolo TASK predict model=MODEL source=SOURCE
```

Examples:

```bash
yolo detect predict model=best.pt source=images/
yolo detect predict model=best.pt source=video.mp4 conf=0.25 save=True
yolo detect predict model=yolo11n.pt source=0
```

## Important official arguments

### Core inference controls

- `source=None`
  - image, video, directory, URL, or live device ID
- `conf=0.25`
  - confidence threshold
- `iou=0.7`
  - NMS IoU threshold
- `imgsz=640`
  - can be int or `(height, width)`
- `rect=True`
  - minimal padding by default
- `half=False`
  - FP16 inference
- `device=None`
- `batch=1`
  - for directory, video, or `.txt` source
- `max_det=300`

### Video and streaming behavior

- `vid_stride=1`
  - process every frame by default
- `stream_buffer=False`
  - `False` drops old frames for lower latency
- `stream=False`
  - memory-efficient generator behavior

### Debugging and advanced behavior

- `visualize=False`
- `augment=False`
- `agnostic_nms=False`
- `classes=None`
- `retina_masks=False`
- `embed=None`
- `compile=False`
- `end2end=None`

### Saved output and visualization

- `show=False`
- `save=False or True`
  - docs say CLI defaults to saved output, Python defaults to unsaved
- `save_frames=False`
- `save_txt=False`
- `save_conf=False`
- `save_crop=False`
- `show_labels=True`
- `show_conf=True`
- `show_boxes=True`
- `line_width=None`
- `project`
- `name`
- `verbose=True`

## Recommended decision rules

- User wants quick visible results:
  - add `save=True`
- User wants downstream processing:
  - add `save_txt=True`, maybe `save_conf=True`
- User wants real-time webcam or RTSP:
  - use `source=0` or stream URL
  - keep `stream_buffer=False` unless frame loss is unacceptable
- User wants higher throughput on folders or video:
  - consider `batch>1`
- User wants lower latency on video:
  - consider `vid_stride>1`

## Expected outputs

Under the company-standard run directory, usually:

- annotated images or videos
- optional per-image text labels
- optional crops

## What to say in the final response

Include:

1. why this is `predict`
2. exact CLI
3. where outputs will appear
4. whether labels, confidences, crops, or rendered media will be saved
