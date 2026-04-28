from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from tests.data_helpers import create_image


def _make_yolo_splits(root: Path, splits: tuple[str, ...]) -> None:
    for split in splits:
        im = root / split / "images"
        im.mkdir(parents=True)
        create_image(im / "sample.png", color=(10, 20, 30), size=(32, 32))
    (root / "classes.txt").write_text("dog\ncat\n", encoding="utf-8")


def test_publish_yolo_dataset_local(client: TestClient, case_dir: Path) -> None:
    ds = case_dir / "dataset"
    project_root = case_dir / "workspace"
    _make_yolo_splits(ds, ("train", "val"))

    r = client.post(
        "/api/v1/preprocess/publish-yolo-dataset",
        json={
            "input_dir": str(ds),
            "project_root_dir": str(project_root),
            "detector_name": "nzxj_louyou",
            "dataset_version": "nzxj_louyou_20260419_1600",
        },
    )
    assert r.status_code == 200
    data = r.json()

    published_dir = project_root / "nzxj_louyou" / "datasets" / "nzxj_louyou_20260419_1600"
    output_yaml = published_dir / "nzxj_louyou_20260419_1600.yaml"
    assert data["publish_mode"] == "local"
    assert data["dataset_version"] == "nzxj_louyou_20260419_1600"
    assert Path(data["published_dataset_dir"]) == published_dir
    assert Path(data["output_yaml_path"]) == output_yaml
    assert data["recommended_train_project"] == str(
        (project_root / "nzxj_louyou" / "runs" / "detect").resolve()
    )
    assert data["recommended_train_name"] == "nzxj_louyou_20260419_1600"
    assert output_yaml.is_file()
    text = output_yaml.read_text(encoding="utf-8")
    assert (published_dir / "train" / "images").resolve().as_posix() in text
    assert (published_dir / "val" / "images").resolve().as_posix() in text


def test_publish_yolo_dataset_remote_sftp(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    from app.services import publish_yolo_dataset as publish_service

    ds = case_dir / "dataset_remote"
    project_root = case_dir / "workspace_remote"
    _make_yolo_splits(ds, ("train", "val"))

    transfer_calls: list[dict] = []
    unzip_calls: list[dict] = []

    def fake_remote_transfer(request):
        transfer_calls.append(request.model_dump())

        class Resp:
            target_path = "/remote/workspace/nzxj_louyou/datasets/nzxj_louyou_20260419_1610.zip"

        return Resp()

    def fake_remote_unzip(request):
        unzip_calls.append(request.model_dump())

        class Resp:
            output_dir = "/remote/workspace/nzxj_louyou/datasets"

        return Resp()

    monkeypatch.setattr(publish_service, "run_remote_transfer", fake_remote_transfer)
    monkeypatch.setattr(publish_service, "run_remote_unzip", fake_remote_unzip)

    r = client.post(
        "/api/v1/preprocess/publish-yolo-dataset",
        json={
            "input_dir": str(ds),
            "project_root_dir": str(project_root),
            "detector_name": "nzxj_louyou",
            "dataset_version": "nzxj_louyou_20260419_1610",
            "publish_mode": "remote_sftp",
            "remote_host": "10.0.0.8",
            "remote_project_root_dir": "/remote/workspace",
            "remote_username": "sk",
            "remote_private_key_path": "/tmp/fake_key",
        },
    )
    assert r.status_code == 200
    data = r.json()

    staging_dir = project_root / "nzxj_louyou" / "datasets" / "nzxj_louyou_20260419_1610"
    staging_yaml = staging_dir / "nzxj_louyou_20260419_1610.yaml"
    assert data["publish_mode"] == "remote_sftp"
    assert data["published_dataset_dir"] == "/remote/workspace/nzxj_louyou/datasets/nzxj_louyou_20260419_1610"
    assert data["output_yaml_path"] == "/remote/workspace/nzxj_louyou/datasets/nzxj_louyou_20260419_1610/nzxj_louyou_20260419_1610.yaml"
    assert data["recommended_train_project"] == "/remote/workspace/nzxj_louyou/runs/detect"
    assert data["recommended_train_name"] == "nzxj_louyou_20260419_1610"
    assert Path(data["staging_published_dataset_dir"]) == staging_dir
    assert Path(data["staging_output_yaml_path"]) == staging_yaml
    assert Path(data["local_archive_path"]).is_file()
    text = staging_yaml.read_text(encoding="utf-8")
    assert "/remote/workspace/nzxj_louyou/datasets/nzxj_louyou_20260419_1610/train/images" in text
    assert str(staging_dir.resolve()) not in text
    assert len(transfer_calls) == 1
    assert len(unzip_calls) == 1
    assert transfer_calls[0]["target"] == "sftp://10.0.0.8/remote/workspace/nzxj_louyou/datasets"
    assert unzip_calls[0]["output_dir"] == "sftp://10.0.0.8/remote/workspace/nzxj_louyou/datasets"


def test_publish_yolo_dataset_local_merges_multiple_inputs_and_last_yaml(
    client: TestClient,
    case_dir: Path,
) -> None:
    ds_a = case_dir / "dataset_a"
    ds_b = case_dir / "dataset_b"
    project_root = case_dir / "workspace_multi"
    _make_yolo_splits(ds_a, ("train",))
    _make_yolo_splits(ds_b, ("val",))
    last_yaml = case_dir / "nzxj_louyou_20260423_2033.yaml"
    last_yaml.write_text(
        "train: /remote/old/train/images\n"
        "val: /remote/old/val/images\n"
        "nc: 2\n"
        "names:\n"
        "  0: dog\n"
        "  1: cat\n",
        encoding="utf-8",
    )

    r = client.post(
        "/api/v1/preprocess/publish-yolo-dataset",
        json={
            "input_dir": str(ds_a),
            "input_dirs": [str(ds_b)],
            "project_root_dir": str(project_root),
            "detector_name": "nzxj_louyou",
            "dataset_version": "nzxj_louyou_20260428_1200",
            "last_yaml": str(last_yaml),
        },
    )
    assert r.status_code == 200
    data = r.json()

    published_dir = project_root / "nzxj_louyou" / "datasets" / "nzxj_louyou_20260428_1200"
    yaml_path = published_dir / "nzxj_louyou_20260428_1200.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    assert data["publish_mode"] == "local"
    assert data["last_yaml_merged"] is True
    assert len(data["source_dataset_roots"]) == 2
    assert "/remote/old/train/images" in text
    assert "/remote/old/val/images" in text
    assert str((published_dir / "dataset_a" / "train" / "images").resolve().as_posix()) in text
    assert str((published_dir / "dataset_b" / "val" / "images").resolve().as_posix()) in text


def test_publish_yolo_dataset_local_auto_discovers_aug_sibling(
    client: TestClient,
    case_dir: Path,
) -> None:
    ds = case_dir / "dataset_auto"
    ds_aug = case_dir / "dataset_auto_aug"
    project_root = case_dir / "workspace_auto"
    _make_yolo_splits(ds, ("train",))
    _make_yolo_splits(ds_aug, ("val",))

    r = client.post(
        "/api/v1/preprocess/publish-yolo-dataset",
        json={
            "input_dir": str(ds),
            "project_root_dir": str(project_root),
            "detector_name": "nzxj_louyou",
            "dataset_version": "nzxj_louyou_20260428_1500",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["source_dataset_roots"]) == 2

    published_dir = project_root / "nzxj_louyou" / "datasets" / "nzxj_louyou_20260428_1500"
    yaml_path = published_dir / "nzxj_louyou_20260428_1500.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    assert str((published_dir / "dataset_auto" / "train" / "images").resolve().as_posix()) in text
    assert str((published_dir / "dataset_auto_aug" / "val" / "images").resolve().as_posix()) in text


def test_publish_yolo_dataset_remote_sftp_uses_env_defaults_and_infers_detector(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    from app.services import publish_yolo_dataset as publish_service

    ds = case_dir / "dataset_remote_env"
    project_root = case_dir / "workspace_remote_env"
    _make_yolo_splits(ds, ("train",))

    transfer_calls: list[dict] = []
    unzip_calls: list[dict] = []

    def fake_remote_transfer(request):
        transfer_calls.append(request.model_dump())

        class Resp:
            target_path = "/remote/workspace/nzxj_louyou/datasets/nzxj_louyou_20260428_1300.zip"

        return Resp()

    def fake_remote_unzip(request):
        unzip_calls.append(request.model_dump())

        class Resp:
            output_dir = "/remote/workspace/nzxj_louyou/datasets"

        return Resp()

    monkeypatch.setenv("SELF_API_REMOTE_SFTP_HOST", "10.0.0.9")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_PROJECT_ROOT_DIR", "/remote/workspace")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_USERNAME", "sk")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_PRIVATE_KEY_PATH", "/tmp/fake_env_key")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_PORT", "2222")
    get_settings.cache_clear()
    monkeypatch.setattr(
        publish_service,
        "_load_last_yaml_text",
        lambda request: (
            "train: /remote/old/train/images\n"
            "nc: 2\n"
            "names:\n"
            "  0: dog\n"
            "  1: cat\n",
            "sftp",
        ),
    )
    monkeypatch.setattr(publish_service, "run_remote_transfer", fake_remote_transfer)
    monkeypatch.setattr(publish_service, "run_remote_unzip", fake_remote_unzip)

    r = client.post(
        "/api/v1/preprocess/publish-yolo-dataset",
        json={
            "input_dir": str(ds),
            "project_root_dir": str(project_root),
            "publish_mode": "remote_sftp",
            "dataset_version": "nzxj_louyou_20260428_1300",
            "last_yaml": "sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260423_2033/nzxj_louyou_20260423_2033.yaml",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["published_dataset_dir"] == "/remote/workspace/nzxj_louyou/datasets/nzxj_louyou_20260428_1300"
    assert data["remote_target_host"] == "10.0.0.9"
    assert data["remote_target_port"] == 2222
    assert transfer_calls[0]["target"] == "sftp://10.0.0.9:2222/remote/workspace/nzxj_louyou/datasets"
    assert transfer_calls[0]["username"] == "sk"
    assert transfer_calls[0]["private_key_path"] == "/tmp/fake_env_key"
    assert unzip_calls[0]["output_dir"] == "sftp://10.0.0.9:2222/remote/workspace/nzxj_louyou/datasets"


def test_publish_yolo_dataset_remote_sftp_infers_fields_from_last_yaml_path(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    from app.services import publish_yolo_dataset as publish_service

    ds = case_dir / "dataset_remote_infer"
    project_root = case_dir / "workspace_remote_infer"
    _make_yolo_splits(ds, ("train",))

    transfer_calls: list[dict] = []
    unzip_calls: list[dict] = []

    def fake_remote_transfer(request):
        transfer_calls.append(request.model_dump())

        class Resp:
            target_path = "/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260428_1400.zip"

        return Resp()

    def fake_remote_unzip(request):
        unzip_calls.append(request.model_dump())

        class Resp:
            output_dir = "/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset"

        return Resp()

    monkeypatch.setenv("SELF_API_REMOTE_SFTP_USERNAME", "sk")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_PRIVATE_KEY_PATH", "/tmp/fake_env_key")
    get_settings.cache_clear()
    monkeypatch.setattr(
        publish_service,
        "_load_last_yaml_text",
        lambda request: (
            "train: /mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260423_2033/train/images\n"
            "nc: 2\n"
            "names:\n"
            "  0: dog\n"
            "  1: cat\n",
            "sftp",
        ),
    )
    monkeypatch.setattr(publish_service, "run_remote_transfer", fake_remote_transfer)
    monkeypatch.setattr(publish_service, "run_remote_unzip", fake_remote_unzip)

    r = client.post(
        "/api/v1/preprocess/publish-yolo-dataset",
        json={
            "input_dir": str(ds),
            "project_root_dir": str(project_root),
            "publish_mode": "remote_sftp",
            "dataset_version": "nzxj_louyou_20260428_1400",
            "last_yaml": "sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260423_2033/nzxj_louyou_20260423_2033.yaml",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["published_dataset_dir"] == "/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260428_1400"
    assert data["output_yaml_path"] == "/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260428_1400/nzxj_louyou_20260428_1400.yaml"
    assert data["remote_target_host"] == "172.31.1.42"
    assert data["remote_target_port"] == 22
    assert transfer_calls[0]["target"] == "sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset"
    assert unzip_calls[0]["output_dir"] == "sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset"


def test_publish_incremental_yolo_dataset_api_minimal_fields(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    from app.services import publish_yolo_dataset as publish_service

    ds = case_dir / "dataset_incremental"
    ds_aug = case_dir / "dataset_incremental_aug"
    _make_yolo_splits(ds, ("train",))
    _make_yolo_splits(ds_aug, ("val",))

    transfer_calls: list[dict] = []
    unzip_calls: list[dict] = []

    def fake_remote_transfer(request):
        transfer_calls.append(request.model_dump())

        class Resp:
            target_path = "/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260428_1600.zip"

        return Resp()

    def fake_remote_unzip(request):
        unzip_calls.append(request.model_dump())

        class Resp:
            output_dir = "/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset"

        return Resp()

    monkeypatch.setenv("SELF_API_REMOTE_SFTP_USERNAME", "sk")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_PRIVATE_KEY_PATH", "/tmp/fake_env_key")
    get_settings.cache_clear()
    monkeypatch.setattr(
        publish_service,
        "_load_last_yaml_text",
        lambda request: (
            "train: /mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260423_2033/train/images\n"
            "nc: 2\n"
            "names:\n"
            "  0: dog\n"
            "  1: cat\n",
            "sftp",
        ),
    )
    monkeypatch.setattr(publish_service, "run_remote_transfer", fake_remote_transfer)
    monkeypatch.setattr(publish_service, "run_remote_unzip", fake_remote_unzip)

    r = client.post(
        "/api/v1/preprocess/publish-incremental-yolo-dataset",
        json={
            "last_yaml": "sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260423_2033/nzxj_louyou_20260423_2033.yaml",
            "local_paths": [str(ds)],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["source_dataset_roots"]) == 2
    assert data["published_dataset_dir"].startswith(
        "/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_"
    )
    assert transfer_calls[0]["target"] == "sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset"
    assert unzip_calls[0]["output_dir"] == "sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset"
