from pathlib import Path

from fastapi.testclient import TestClient

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
