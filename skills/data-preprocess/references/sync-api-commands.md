# Sync API Commands

The commands below are the primary synchronous examples extracted from `docs/api_examples.md`.

Replace paths and host values as needed.

## Discover leaf dirs

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/discover-leaf-dirs" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/raw_nested_dataset"
  }'
```

## Clean nested dataset

### Keep original structure

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/clean-nested-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/raw_nested_dataset",
    "output_dir": "/path/to/cleaned_dataset"
  }'
```

### Flatten to one `images/` + `xmls/` dataset

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/clean-nested-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/raw_nested_dataset",
    "output_dir": "/path/to/cleaned_flat",
    "recursive": true,
    "pairing_mode": "images_xmls_subfolders",
    "flatten": true,
    "include_backgrounds": false,
    "images_dir_name": "images",
    "xmls_dir_name": "xmls",
    "include_difficult": false,
    "copy_files": true,
    "overwrite": true
  }'
```

## Aggregate nested dataset

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/aggregate-nested-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/cleaned_dataset",
    "output_dir": "/path/to/dataset"
  }'
```

## XML to YOLO

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/xml-to-yolo" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/voc_like_dataset"
  }'
```

## Annotate visualize

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/annotate-visualize" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/dataset",
    "output_dir": "/path/to/dataset/visualized"
  }'
```

## Reset YOLO label index to 0

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/reset-yolo-label-index" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/dataset"
  }'
```

## Split YOLO dataset

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/split-yolo-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/yolo_raw",
    "output_dir": "/path/to/yolo_split"
  }'
```

## YOLO augment

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/yolo-augment" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/dataset"
  }'
```

## YOLO sliding-window crop

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/yolo-sliding-window-crop" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/dataset",
    "output_dir": "/path/to/dataset/data_crops"
  }'
```

## Build YOLO YAML

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/build-yolo-yaml" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/temp_dataset",
    "output_yaml_path": "/path/to/temp_dataset/dataset.yaml"
  }'
```

## Zip folder

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/zip-folder" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/source_folder",
    "output_zip_path": "/path/to/source_folder.zip"
  }'
```

## Remote transfer

### Password auth

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/remote-transfer" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "/path/to/dataset",
    "target": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/",
    "username": "sk",
    "password": "your_password",
    "overwrite": true
  }'
```

### Private key auth

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/remote-transfer" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "/path/to/dataset",
    "target": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/",
    "username": "sk",
    "private_key_path": "~/.ssh/id_rsa"
  }'
```

## Remote unzip

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/remote-unzip" \
  -H "Content-Type: application/json" \
  -d '{
    "archive_path": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/dataset.zip",
    "output_dir": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/dataset",
    "username": "sk",
    "private_key_path": "~/.ssh/id_ed25519"
  }'
```

## Unzip archive

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/unzip-archive" \
  -H "Content-Type: application/json" \
  -d '{
    "archive_path": "/path/to/source_folder.zip",
    "output_dir": "/path/to/unpacked"
  }'
```

## Move path

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/move-path" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "/path/to/unpacked",
    "target_dir": "/path/to/archive_ready"
  }'
```

## Copy path

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/copy-path" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "/path/to/unpacked",
    "target_dir": "/path/to/archive_backup"
  }'
```
