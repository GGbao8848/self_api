# Common Chains

Choose the shortest chain that matches the user's dataset layout and end goal.

## Small-image baseline training prep

Use when the input is already `images+xmls` and the goal is standard small-image training prep.

1. `xml-to-yolo`
2. `split-yolo-dataset`
3. `yolo-augment` if augmentation is needed
4. `build-yolo-yaml`
5. `zip-folder` or `remote-transfer` / `remote-unzip`
6. Hand off to `$ultralytics-yolo-modes` for training-stage parameter normalization
7. Local execution injects `yolo-train`; remote execution injects `remote-sbatch-yolo-train`

## Large-image frequent iteration

Use when the source is large images and training depends on sliding-window crops.

1. `clean-nested-dataset` if raw data is still scattered
2. `xml-to-yolo`
3. `reset-yolo-label-index` for single-class tasks
4. `yolo-sliding-window-crop`
5. `split-yolo-dataset` or directly `build-yolo-yaml`
6. `zip-folder` or `remote-transfer` / `remote-unzip`
7. Hand off to `$ultralytics-yolo-modes` for training-stage parameter normalization
8. Local execution injects `yolo-train`; remote execution injects `remote-sbatch-yolo-train`

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
