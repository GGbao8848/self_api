# High-Frequency SOPs

These are the primary workflows used by this API project.

This file is for combined-workflow SOP selection only.

After choosing an SOP:

- route data-processing steps to `$data-preprocess`
- route model CLI steps to `$ultralytics-yolo-modes`

If the user needs only one of those stages, do not use these long SOPs.

## SOP 1: Small-image `images+xmls` baseline training prep

Use when:

- the source is already a standard small-image dataset
- labels are VOC XML
- the goal is baseline model training prep

Sequence:

1. `xml-to-yolo`
2. `split-yolo-dataset`
3. `yolo-augment` if augmentation is needed
4. `publish-yolo-dataset` (lands the versioned dataset and emits `<version>.yaml` in one call; supports `publish_mode=local` or `remote_sftp` to push + unzip remotely)
5. Route the preprocessing stage to `$data-preprocess`
6. Route the training stage to `$ultralytics-yolo-modes`
7. If training is local, inject `yolo-train`
8. If training is remote, inject `remote-sbatch-yolo-train`

Key decisions:

- Skip `yolo-augment` if the dataset is already large enough or the user wants a clean baseline.
- Use `publish_mode=remote_sftp` instead of separate `zip-folder` + `remote-transfer` + `remote-unzip` when delivery + version pinning should be one step.
- `build-yolo-yaml` (removed) is now subsumed by `publish-yolo-dataset`.

## SOP 2: Large-image `images+xmls` frequent iteration

Use when:

- source data may still be scattered
- training relies on sliding-window crops
- iteration speed matters more than perfect one-shot cleanup

Sequence:

1. `clean-nested-dataset` if the original directory is messy
2. `xml-to-yolo`
3. `reset-yolo-label-index` for single-class training
4. `split-yolo-dataset` (split the original large images into train/val first to prevent data leakage)
5. `yolo-sliding-window-crop` with `input_dir=<split_dir>` so train and val crops stay in their own splits
6. `yolo-augment` on `<split_dir>/crop/train` if augmentation is needed
7. `publish-yolo-dataset` with `input_dir=<split_dir>/crop` (publishes the small-image + augmentation dataset version and emits yaml; choose `publish_mode=local` or `remote_sftp`)
8. Route the preprocessing stage to `$data-preprocess`
9. Route the training stage to `$ultralytics-yolo-modes`
10. If training is local, inject `yolo-train`
11. If training is remote, inject `remote-sbatch-yolo-train`

Key decisions:

- Use `clean-nested-dataset` with `flatten: true` when you want one merged crop source dataset.
- Cropping **after** `split-yolo-dataset` prevents the same source image contributing patches to both train and val.
- `build-yolo-yaml` (removed) is now subsumed by `publish-yolo-dataset`.

## SOP 3: Multi-layer dataset consolidation

Use when:

- original data is distributed across many branches
- the first problem is understanding and cleaning the directory tree

Sequence:

1. `discover-leaf-dirs`
2. `clean-nested-dataset`
3. `xml-to-yolo`
4. `aggregate-nested-dataset`
5. Route execution to `$data-preprocess`

Key decisions:

- Use this SOP before any train-oriented SOP when the source structure is still unclear.
- Prefer `pairing_mode: "images_xmls_subfolders"` for VOC-style sample folders.

## SOP 4: Dataset delivery to another machine

Use when:

- preprocessing is complete
- the next step is remote training or remote staging

Sequence:

1. `zip-folder`
2. `remote-transfer`
3. `remote-unzip`
4. Route transfer execution to `$data-preprocess`
5. Route training-stage normalization to `$ultralytics-yolo-modes` when needed
6. If the next step is remote cluster execution, inject `remote-sbatch-yolo-train`

Local-only variants:

- `unzip-archive`
- `move-path`
- `copy-path`

## SOP boundaries

This skill chooses the workflow and decides the handoff point.

Execution belongs to:

- `$data-preprocess` for data tasks
- `$ultralytics-yolo-modes` for model CLI tasks
