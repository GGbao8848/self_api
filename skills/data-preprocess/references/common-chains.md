# Common Chains

Choose the shortest chain that matches the user's dataset layout and end goal.

## Small-image baseline training prep

Use when the input is already `images+xmls` and the goal is standard small-image training prep.

1. `xml-to-yolo`
2. `split-yolo-dataset`
3. `yolo-augment` if augmentation is needed
4. `publish-yolo-dataset` (lands the versioned dataset and emits `<version>.yaml` in one call; supports `publish_mode=local` or `remote_sftp`)
5. Hand off to `$ultralytics-yolo-modes` for training-stage parameter normalization
6. Local execution injects `yolo-train`; remote execution injects `remote-sbatch-yolo-train`

> `build-yolo-yaml` has been removed; `publish-yolo-dataset` builds the yaml internally.

## Large-image frequent iteration

Use when the source is large images and training depends on sliding-window crops.
To avoid data leakage, split first, then crop each split independently.

1. `clean-nested-dataset` if raw data is still scattered
2. `xml-to-yolo`
3. `reset-yolo-label-index` for single-class tasks
4. `split-yolo-dataset` (split the original large images into train/val first)
5. `yolo-sliding-window-crop` with `input_dir=<split_dir>` so train and val get cropped separately
6. `yolo-augment` on `<split_dir>/crop/train` if augmentation is needed
7. `publish-yolo-dataset` with `input_dir=<split_dir>/crop` (publishes the small-image + augmentation dataset version and emits yaml)
8. Hand off to `$ultralytics-yolo-modes` for training-stage parameter normalization
9. Local execution injects `yolo-train`; remote execution injects `remote-sbatch-yolo-train`

## Multi-layer directory consolidation

Use when the input has many nested folders and the user first needs dataset cleanup.

1. `discover-leaf-dirs`
2. `clean-nested-dataset`
3. `xml-to-yolo`
4. `aggregate-nested-dataset`

## Delivery and cross-server transfer

Use when the local dataset is already prepared and the main task is shipping it.

1. `zip-folder` when archive delivery is preferred
2. `remote-transfer`
3. `remote-unzip`

If the user wants local unpacking or staging only, use `unzip-archive`, `move-path`, and `copy-path`.
