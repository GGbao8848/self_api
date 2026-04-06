from pathlib import Path

from fastapi.testclient import TestClient

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
