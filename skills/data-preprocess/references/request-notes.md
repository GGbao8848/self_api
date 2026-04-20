# Request Notes

These are the request-shaping rules that matter most when producing commands.

## Directory assumptions

- `xml-to-yolo` expects a dataset root with `images/` and `xmls/`.
- `annotate-visualize` expects `input_dir/images`; it prefers `input_dir/labels`, otherwise falls back to `input_dir/xmls`.
- `reset-yolo-label-index` works in place on `input_dir/labels`.
- `split-yolo-dataset` expects `input_dir/images` and `input_dir/labels`.
- `yolo-augment` expects YOLO TXT under `images/` and `labels/`.
- `yolo-sliding-window-crop` expects `images/` and optionally `labels/`.

## Important parameter choices

### `clean-nested-dataset`

- Standard mode: pass only `input_dir` and `output_dir`.
- Use `pairing_mode: "same_directory"` when images and XML files live together in leaf folders.
- Use `pairing_mode: "images_xmls_subfolders"` when each sample folder contains sibling `images/` and `xmls/` subfolders.
- Use `flatten: true` when the user wants a single merged `images/` + `xmls/` output.
- Use `include_backgrounds: false` when only labeled pairs should be kept.

### `publish-yolo-dataset` (replaces removed `build-yolo-yaml`)

- As of 2026-04, `POST /api/v1/preprocess/build-yolo-yaml` and its async variant are
  **removed**. Use `POST /api/v1/preprocess/publish-yolo-dataset` instead — it lands the
  dataset version at `<project_root_dir>/<detector_name>/datasets/<version>/` and
  **emits `<version>.yaml` automatically** (train/val/test `images` absolute paths plus
  `nc`/`names`).
- If `classes.txt` is empty or missing, pass `last_yaml` so class names come from there.
- `publish_mode="local"` writes on-disk only; `publish_mode="remote_sftp"` additionally
  zips, SFTPs to `remote_host`, and unzips remotely.

### Transfer endpoints

- `remote-transfer` supports `sftp://host/path`, `sftp://user@host/path`, `sftp://host:port/path`, and `user@host:path`.
- `remote-unzip` executes unzip on the target host and is the preferred follow-up after transferring archives.

## Synchronous-first rule

Default to synchronous examples from this skill unless the user explicitly requests:

- async callbacks
- task polling
- LangGraph pipeline orchestration (`/api/v1/pipeline/*`)
- long-running remote jobs
