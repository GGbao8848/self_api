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


def test_build_yolo_yaml_sync(client: TestClient, case_dir: Path) -> None:
    ds = case_dir / "dataset"
    _make_yolo_splits(ds, ("train", "val", "test"))
    out = ds / "dataset.yaml"
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={
            "input_dir": str(ds),
            "output_yaml_path": str(out),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["splits_included"] == ["train", "val", "test"]
    assert data["classes_count"] == 2
    assert Path(data["output_yaml_path"]) == out
    train_abs = (ds / "train" / "images").resolve().as_posix()
    val_abs = (ds / "val" / "images").resolve().as_posix()
    test_abs = (ds / "test" / "images").resolve().as_posix()
    text = out.read_text(encoding="utf-8")
    assert f"train: {train_abs}" in text
    assert f"val: {val_abs}" in text
    assert f"test: {test_abs}" in text
    assert data["path_in_yaml"] == train_abs
    assert "nc: 2" in text
    assert "0: dog" in text


def test_build_yolo_yaml_path_prefix_replace(client: TestClient, case_dir: Path) -> None:
    ds = case_dir / "dataset2"
    _make_yolo_splits(ds, ("train", "val", "test"))
    out = ds / "remote.yaml"
    resolved = ds.resolve().as_posix()
    train_abs = (ds / "train" / "images").resolve().as_posix()
    expected_train = "/mnt/training/dataset2" + train_abs[len(resolved) :]
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={
            "input_dir": str(ds),
            "path_prefix_replace_from": resolved,
            "path_prefix_replace_to": "/mnt/training/dataset2",
            "output_yaml_path": str(out),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["path_in_yaml"] == expected_train
    assert f"train: {expected_train}" in out.read_text(encoding="utf-8")


def test_build_yolo_yaml_path_prefix_replace_no_double_slash(
    client: TestClient, case_dir: Path
) -> None:
    """path_prefix_replace_to 带尾部 / 时不应出现 n8n_workspace//dataset 式双斜杠。"""
    ds = case_dir / "dataset2b"
    _make_yolo_splits(ds, ("train", "val", "test"))
    out = ds / "remote2.yaml"
    resolved = ds.resolve().as_posix()
    train_abs = (ds / "train" / "images").resolve().as_posix()
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={
            "input_dir": str(ds),
            "path_prefix_replace_from": resolved,
            "path_prefix_replace_to": "/mnt/training/dataset2b/",
            "output_yaml_path": str(out),
        },
    )
    assert r.status_code == 200
    expected_train = (Path("/mnt/training/dataset2b") / "train" / "images").as_posix()
    assert r.json()["path_in_yaml"] == expected_train
    text = out.read_text(encoding="utf-8")
    assert f"train: {expected_train}" in text
    assert "//" not in text.split("train:", 1)[1].split("\n", 1)[0]


def test_build_yolo_yaml_train_only(client: TestClient, case_dir: Path) -> None:
    ds = case_dir / "dataset3"
    _make_yolo_splits(ds, ("train",))
    out = ds / "t.yaml"
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={"input_dir": str(ds), "output_yaml_path": str(out)},
    )
    assert r.status_code == 200
    assert r.json()["splits_included"] == ["train"]
    assert "val:" not in out.read_text(encoding="utf-8")


def test_build_yolo_yaml_auto_yolo_split_images_first(
    client: TestClient, case_dir: Path
) -> None:
    """original_dataset-style root: yolo_split/images/<split>/ (split-yolo-dataset output)."""
    root = case_dir / "project"
    ys = root / "yolo_split"
    for split in ("train", "val", "test"):
        d = ys / "images" / split
        d.mkdir(parents=True)
        create_image(d / "a.png", color=(1, 2, 3), size=(16, 16))
    (ys / "classes.txt").write_text("dog\ncat\n", encoding="utf-8")
    out = root / "data.yaml"
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={"input_dir": str(root), "output_yaml_path": str(out)},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["dataset_root"].endswith("yolo_split")
    train_abs = (ys / "images" / "train").resolve().as_posix()
    text = out.read_text(encoding="utf-8")
    assert f"train: {train_abs}" in text


def test_build_yolo_yaml_auto_dataset_subfolder(
    client: TestClient, case_dir: Path
) -> None:
    """original_dataset + dataset/ (cropped temp_dataset) with train/images."""
    root = case_dir / "project2"
    ds = root / "dataset"
    for split in ("train", "val"):
        im = ds / split / "images"
        im.mkdir(parents=True)
        create_image(im / "x.png", color=(5, 5, 5), size=(8, 8))
    (ds / "classes.txt").write_text("dog\n", encoding="utf-8")
    out = root / "out.yaml"
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={"input_dir": str(root), "output_yaml_path": str(out)},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["dataset_root"].endswith("dataset")
    train_abs = (ds / "train" / "images").resolve().as_posix()
    assert f"train: {train_abs}" in out.read_text(encoding="utf-8")


def test_build_yolo_yaml_multi_image_dirs_per_split(
    client: TestClient, case_dir: Path
) -> None:
    ds = case_dir / "dataset_multi"
    for split in ("train", "val"):
        base_images = ds / split / "images"
        aug_images = ds / split / "augment" / "images"
        base_images.mkdir(parents=True)
        aug_images.mkdir(parents=True)
        create_image(base_images / "a.png", color=(1, 2, 3), size=(16, 16))
        create_image(aug_images / "b.png", color=(4, 5, 6), size=(16, 16))
    (ds / "classes.txt").write_text("dog\n", encoding="utf-8")
    out = ds / "multi.yaml"

    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={"input_dir": str(ds), "output_yaml_path": str(out)},
    )
    assert r.status_code == 200

    text = out.read_text(encoding="utf-8")
    train_main = (ds / "train" / "images").resolve().as_posix()
    train_aug = (ds / "train" / "augment" / "images").resolve().as_posix()
    val_main = (ds / "val" / "images").resolve().as_posix()
    val_aug = (ds / "val" / "augment" / "images").resolve().as_posix()
    assert "train:" in text
    assert f"  - {train_main}" in text
    assert f"  - {train_aug}" in text
    assert "val:" in text
    assert f"  - {val_main}" in text
    assert f"  - {val_aug}" in text


def test_build_yolo_yaml_supports_split_anchor_under_named_parent(
    client: TestClient,
    case_dir: Path,
) -> None:
    ds = case_dir / "crops"
    for split in ("train", "val"):
        base_images = ds / split / "images"
        aug_images = ds / "augment" / split / "images"
        base_images.mkdir(parents=True)
        aug_images.mkdir(parents=True)
        create_image(base_images / "a.png", color=(1, 2, 3), size=(16, 16))
        create_image(aug_images / "b.png", color=(4, 5, 6), size=(16, 16))
    (ds / "classes.txt").write_text("dog\n", encoding="utf-8")
    out = ds / "nested_split_anchor.yaml"

    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={"input_dir": str(ds), "output_yaml_path": str(out)},
    )
    assert r.status_code == 200

    text = out.read_text(encoding="utf-8")
    train_main = (ds / "train" / "images").resolve().as_posix()
    train_aug = (ds / "augment" / "train" / "images").resolve().as_posix()
    val_main = (ds / "val" / "images").resolve().as_posix()
    val_aug = (ds / "augment" / "val" / "images").resolve().as_posix()
    assert "train:" in text
    assert f"  - {train_main}" in text
    assert f"  - {train_aug}" in text
    assert "val:" in text
    assert f"  - {val_main}" in text
    assert f"  - {val_aug}" in text


def test_build_yolo_yaml_async(client: TestClient, case_dir: Path) -> None:
    import time

    ds = case_dir / "dataset4"
    _make_yolo_splits(ds, ("train", "val", "test"))
    out = ds / "a.yaml"
    submit = client.post(
        "/api/v1/preprocess/build-yolo-yaml/async",
        json={"input_dir": str(ds), "output_yaml_path": str(out)},
    )
    assert submit.status_code == 202
    tid = submit.json()["task_id"]
    deadline = time.time() + 5.0
    while time.time() < deadline:
        st = client.get(f"/api/v1/preprocess/tasks/{tid}")
        assert st.status_code == 200
        body = st.json()
        if body["state"] in {"succeeded", "failed"}:
            assert body["state"] == "succeeded"
            assert body["result"]["splits_included"] == ["train", "val", "test"]
            assert out.is_file()
            train_abs = (ds / "train" / "images").resolve().as_posix()
            assert f"train: {train_abs}" in out.read_text(encoding="utf-8")
            return
        time.sleep(0.05)
    raise AssertionError("async build-yolo-yaml did not finish")


def test_build_yolo_yaml_merge_last_local(client: TestClient, case_dir: Path) -> None:
    ds = case_dir / "dataset_merge"
    _make_yolo_splits(ds, ("train", "val", "test"))
    prev = ds / "previous.yaml"
    prev.write_text(
        "train: /legacy/train/images\n"
        "val: /legacy/val/images\n"
        "nc: 99\n"
        "names:\n"
        "  0: ignore\n",
        encoding="utf-8",
    )
    out = ds / "merged.yaml"
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={
            "input_dir": str(ds),
            "last_yaml": str(prev),
            "output_yaml_path": str(out),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["last_yaml_merged"] is True
    assert data["last_yaml_source"] == "local"
    assert data["classes_count"] == 2
    text = out.read_text(encoding="utf-8")
    train_new = (ds / "train" / "images").resolve().as_posix()
    assert "train:" in text
    assert "/legacy/train/images" in text
    assert train_new in text
    assert "nc: 2" in text
    assert "0: dog" in text


def test_build_yolo_yaml_empty_classes_txt_uses_last_yaml_names(
    client: TestClient, case_dir: Path,
) -> None:
    ds = case_dir / "dataset_empty_cls"
    _make_yolo_splits(ds, ("train", "val", "test"))
    (ds / "classes.txt").write_text("# no lines\n", encoding="utf-8")
    base = ds / "previous.yaml"
    base.write_text(
        "train: /legacy/train/images\n"
        "val: /legacy/val/images\n"
        "nc: 2\n"
        "names:\n"
        "  0: dog\n"
        "  1: cat\n",
        encoding="utf-8",
    )
    out = ds / "out_empty_cls.yaml"
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={
            "input_dir": str(ds),
            "last_yaml": str(base),
            "output_yaml_path": str(out),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["classes_count"] == 2
    text = out.read_text(encoding="utf-8")
    train_new = (ds / "train" / "images").resolve().as_posix()
    assert "/legacy/train/images" in text
    assert train_new in text
    assert "nc: 2" in text
    assert "0: dog" in text


def test_build_yolo_yaml_no_classes_file_uses_last_yaml(
    client: TestClient, case_dir: Path,
) -> None:
    ds = case_dir / "dataset_no_cls_file"
    for split in ("train", "val"):
        im = ds / split / "images"
        im.mkdir(parents=True)
        create_image(im / "sample.png", color=(10, 20, 30), size=(32, 32))
    base = ds / "base.yaml"
    base.write_text(
        "train: /a/images\n"
        "val: /b/images\n"
        "names:\n"
        "  0: only\n",
        encoding="utf-8",
    )
    out = ds / "merged_no_cls.yaml"
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={
            "input_dir": str(ds),
            "last_yaml": str(base),
            "output_yaml_path": str(out),
        },
    )
    assert r.status_code == 200
    assert r.json()["classes_count"] == 1
    text = out.read_text(encoding="utf-8")
    assert "0: only" in text
    assert (ds / "train" / "images").resolve().as_posix() in text


def test_build_yolo_yaml_empty_classes_without_last_yaml_fails(
    client: TestClient, case_dir: Path,
) -> None:
    ds = case_dir / "dataset_empty_fail"
    _make_yolo_splits(ds, ("train", "val", "test"))
    (ds / "classes.txt").write_text("\n", encoding="utf-8")
    out = ds / "fail.yaml"
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={"input_dir": str(ds), "output_yaml_path": str(out)},
    )
    assert r.status_code == 400


def test_build_yolo_yaml_last_yaml_remote_requires_sftp_fields(
    client: TestClient, case_dir: Path,
) -> None:
    ds = case_dir / "dataset_sftp_err"
    _make_yolo_splits(ds, ("train", "val", "test"))
    out = ds / "out.yaml"
    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={
            "input_dir": str(ds),
            "last_yaml": "sftp://example.com/var/data.yaml",
            "output_yaml_path": str(out),
        },
    )
    assert r.status_code == 400
    assert "sftp_username" in r.json()["detail"]


def test_build_yolo_yaml_last_yaml_remote_uses_env_defaults(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    from app.services import build_yolo_yaml as build_service

    ds = case_dir / "dataset_sftp_env"
    _make_yolo_splits(ds, ("train", "val"))
    out = ds / "out.yaml"

    monkeypatch.setenv("SELF_API_REMOTE_SFTP_USERNAME", "sk")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_PRIVATE_KEY_PATH", "/tmp/fake_env_key")
    monkeypatch.setenv("SELF_API_REMOTE_SFTP_PORT", "2222")
    get_settings.cache_clear()

    def fake_read_sftp_file_text(target, username, private_key_path, port=None):
        assert target == "sftp://example.com/var/data.yaml"
        assert username == "sk"
        assert private_key_path == "/tmp/fake_env_key"
        assert port == 2222
        return "train: /remote/old/train/images\nnc: 2\nnames:\n  0: dog\n  1: cat\n"

    monkeypatch.setattr(build_service, "read_sftp_file_text", fake_read_sftp_file_text)

    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={
            "input_dir": str(ds),
            "last_yaml": "sftp://example.com/var/data.yaml",
            "output_yaml_path": str(out),
        },
    )
    assert r.status_code == 200
    assert r.json()["last_yaml_source"] == "sftp"


def test_build_yolo_yaml_can_publish_dataset_into_project_structure(
    client: TestClient,
    case_dir: Path,
) -> None:
    source_root = case_dir / "tmp_dataset" / "crops"
    project_root = case_dir / "workspace_root" / "TVDS"
    for split in ("train", "val"):
        base_images = source_root / split / "images"
        aug_images = source_root / "augment" / split / "images"
        base_labels = source_root / split / "labels"
        aug_labels = source_root / "augment" / split / "labels"
        base_images.mkdir(parents=True)
        aug_images.mkdir(parents=True)
        base_labels.mkdir(parents=True)
        aug_labels.mkdir(parents=True)
        create_image(base_images / "a.png", color=(1, 2, 3), size=(16, 16))
        create_image(aug_images / "b.png", color=(4, 5, 6), size=(16, 16))
        (base_labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (aug_labels / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    classes_file = source_root.parent / "classes.txt"
    classes_file.write_text("dog\n", encoding="utf-8")

    r = client.post(
        "/api/v1/preprocess/build-yolo-yaml",
        json={
            "input_dir": str(source_root),
            "project_root_dir": str(project_root),
            "detector_name": "nzxj_louyou",
            "dataset_version": "nzxj_louyou_20260419_1510",
        },
    )
    assert r.status_code == 200
    data = r.json()

    published_dir = (
        project_root / "nzxj_louyou" / "datasets" / "nzxj_louyou_20260419_1510"
    ).resolve()
    yaml_path = published_dir / "nzxj_louyou_20260419_1510.yaml"
    text = yaml_path.read_text(encoding="utf-8")

    assert data["dataset_version"] == "nzxj_louyou_20260419_1510"
    assert Path(data["published_dataset_dir"]) == published_dir
    assert Path(data["output_yaml_path"]) == yaml_path
    assert data["recommended_train_project"] == str(
        (project_root / "nzxj_louyou" / "runs" / "detect").resolve()
    )
    assert data["recommended_train_name"] == "nzxj_louyou_20260419_1510"
    assert (published_dir / "classes.txt").read_text(encoding="utf-8").strip() == "dog"

    train_main = (published_dir / "train" / "images").resolve().as_posix()
    train_aug = (published_dir / "augment" / "train" / "images").resolve().as_posix()
    val_main = (published_dir / "val" / "images").resolve().as_posix()
    val_aug = (published_dir / "augment" / "val" / "images").resolve().as_posix()
    assert f"  - {train_main}" in text
    assert f"  - {train_aug}" in text
    assert f"  - {val_main}" in text
    assert f"  - {val_aug}" in text
