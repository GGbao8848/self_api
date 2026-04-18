# Train Mode

Official source: `https://docs.ultralytics.com/modes/train/`
Compatibility baseline: `ultralytics v8.4.38`

Use this file when the user wants to train or continue training a YOLO model.

For company path, naming, bucket, and questioning rules, follow [company-cli-standard.md](company-cli-standard.md). If anything in this file appears to conflict with company conventions, the company standard wins.

## What train mode is for

The official docs describe train mode as the mode for fitting a YOLO model on a custom dataset, with support for:

- pretrained `.pt` weights
- model `.yaml` definitions
- multi-GPU training
- idle GPU auto-selection
- Apple silicon MPS
- resume from interrupted runs

## Minimum inputs

- `root_dir`: business project root
- `detector_name`: detector or model name
- `TASK`: usually `detect`, `segment`, `classify`, `pose`, or `obb`
- `model=...`
- `data=...`

Derived by company standard:

- `project=...`
- `name=...`

If the user only says "训练一个模型", do not stop at that. Resolve:

- `root_dir`
- `detector_name`
- dataset YAML path
- task type
- whether to start from pretrained `.pt` or architecture `.yaml`
- enough information to derive `project`
- enough information to derive `name`

## Canonical commands

Start from pretrained weights:

```bash
yolo TASK train model=yolo11n.pt data=DATA_YAML project=PROJECT name=NAME epochs=100 imgsz=640
```

Train from architecture YAML:

```bash
yolo TASK train model=yolo11n.yaml data=DATA_YAML project=PROJECT name=NAME epochs=100 imgsz=640
```

Transfer pretrained weights into a YAML-defined architecture:

```bash
yolo TASK train model=yolo11n.yaml pretrained=yolo11n.pt data=DATA_YAML project=PROJECT name=NAME epochs=100 imgsz=640
```

Resume interrupted training:

```bash
yolo train resume model=PATH_TO_LAST_PT
```

Multi-GPU:

```bash
yolo TASK train model=yolo11n.pt data=DATA_YAML project=PROJECT name=NAME epochs=100 imgsz=640 device=0,1
```

Use most idle GPU:

```bash
yolo TASK train model=yolo11n.pt data=DATA_YAML project=PROJECT name=NAME epochs=100 imgsz=640 device=-1
```

Apple silicon:

```bash
yolo TASK train model=yolo11n.pt data=DATA_YAML project=PROJECT name=NAME epochs=100 imgsz=640 device=mps
```

## Important official arguments

The official train page lists many arguments. These are the ones most likely to matter for skill decisions.

### Required or near-required

- `model`
  - `.pt` means pretrained start
  - `.yaml` means define architecture from scratch
- `data`
  - dataset YAML containing paths, class names, class count

### Core optimization controls

- `epochs=100`
  - total epochs
- `time=None`
  - max training hours; overrides `epochs`
- `patience=100`
  - early stopping if validation stops improving
- `batch=16`
  - can be integer, `-1`, or fraction like `0.70`
- `imgsz=640`
  - target training image size
- `optimizer='auto'`
  - official docs say `auto` may choose MuSGD for longer runs, AdamW for shorter runs
- `lr0=0.01`
  - initial learning rate

### Additional official training arguments commonly used

- `freeze=None`
  - freeze the first N layers or a list of layer indices
- `lrf=0.01`
  - final learning rate fraction
- `momentum=0.937`
- `weight_decay=0.0005`
- `warmup_epochs=3.0`
- `warmup_momentum=0.8`
- `warmup_bias_lr=0.1`
- `box=7.5`
- `cls=0.5`
- `cls_pw=0.0`
- `dfl=1.5`
- `pose=12.0`
- `kobj=1.0`
- `rle=1.0`
- `angle=1.0`
- `nbs=64`
- `profile=False`
- `plots=True`
- `max_det=300`

### Runtime and hardware

- `device=None`
  - auto-select GPU 0 if available, else CPU
  - can also be `0`, `0,1`, `cpu`, `mps`, `npu`, `-1`
- `workers=8`
  - data loader workers
- `amp=True`
  - automatic mixed precision

### Experiment management

- `project`
- `name`
- `exist_ok=False`
- `save=True`
- `save_period=-1`
- `resume=False`

### Dataset shaping and transfer learning

- `cache=False`
  - can be `True`, `ram`, `disk`, or `False`
- `pretrained=True`
  - bool or checkpoint path
- `single_cls=False`
- `classes=None`
- `freeze=None`
- `fraction=1.0`
- `rect=False`
- `multi_scale=0.0`

### Official augmentation arguments

- `hsv_h=0.015`
- `hsv_s=0.7`
- `hsv_v=0.4`
- `degrees=0.0`
- `translate=0.1`
- `scale=0.5`
- `shear=0.0`
- `perspective=0.0`
- `flipud=0.0`
- `fliplr=0.5`
- `bgr=0.0`
- `mosaic=1.0`
- `mixup=0.0`
- `cutmix=0.0`
- `copy_paste=0.0`
- `copy_paste_mode='flip'`
- `auto_augment='randaugment'`
- `erasing=0.4`

### Stability and schedule

- `seed=0`
- `deterministic=True`
- `cos_lr=False`
- `close_mosaic=10`

## Recommended decision rules

- User wants fastest smoke test:
  - use a nano or small pretrained checkpoint
  - keep `imgsz=640`
  - use modest `epochs`
- User wants highest practical accuracy:
  - keep pretrained start unless they explicitly want from-scratch
  - ask about compute budget only if it materially changes command choice
- User has shared multi-user GPU machine:
  - prefer `device=-1` for low coordination friction
- User wants to continue a failed run:
  - prefer `yolo train resume model=.../last.pt`

## Expected outputs

Under the company-standard run directory, typically:

- `weights/best.pt`
- `weights/last.pt`
- training plots and metrics artifacts

## What to say in the final response

Include:

1. inferred task and why it is `train`
2. exact CLI
3. expected output directory and key artifact, usually `best.pt`
4. one next step such as validate or export
