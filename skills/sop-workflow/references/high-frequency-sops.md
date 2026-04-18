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
4. `build-yolo-yaml`
5. `zip-folder` or `remote-transfer -> remote-unzip` if the dataset must move
6. Route the preprocessing stage to `$data-preprocess`
7. Route the training stage to `$ultralytics-yolo-modes`
8. If training is local, inject `yolo-train`
9. If training is remote, inject `remote-sbatch-yolo-train`

Key decisions:

- Skip `yolo-augment` if the dataset is already large enough or the user wants a clean baseline.
- Use `remote-transfer` directly only when archive packaging is unnecessary.

## SOP 2: Large-image `images+xmls` frequent iteration

Use when:

- source data may still be scattered
- training relies on sliding-window crops
- iteration speed matters more than perfect one-shot cleanup

Sequence:

1. `clean-nested-dataset` if the original directory is messy
2. `xml-to-yolo`
3. `reset-yolo-label-index` for single-class training
4. `yolo-sliding-window-crop`
5. `split-yolo-dataset` or directly `build-yolo-yaml`
6. `zip-folder` or `remote-transfer -> remote-unzip`
7. Route the preprocessing stage to `$data-preprocess`
8. Route the training stage to `$ultralytics-yolo-modes`
9. If training is local, inject `yolo-train`
10. If training is remote, inject `remote-sbatch-yolo-train`

Key decisions:

- Use `clean-nested-dataset` with `flatten: true` when you want one merged crop source dataset.
- If training consumes the crop dataset directly without split, go straight to `build-yolo-yaml`.

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
