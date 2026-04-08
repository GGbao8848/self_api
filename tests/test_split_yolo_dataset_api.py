from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_yolo_dataset


def test_split_yolo_dataset_endpoint_train_val_test(
    client: TestClient,
    case_dir: Path,
) -> None:
    dataset_dir = case_dir / "yolo_dataset"
    output_dir = case_dir / "split_out"
    create_yolo_dataset(dataset_dir, sample_count=8)

    response = client.post(
        "/api/v1/preprocess/split-yolo-dataset",
        json={
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "mode": "train_val_test",
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "shuffle": False,
            "output_layout": "images_first",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["total_images"] == 8
    assert data["paired_images"] == 8
    assert data["skipped_images"] == 0
    assert data["train_images"] == 4
    assert data["val_images"] == 2
    assert data["test_images"] == 2
    assert len(list((output_dir / "images" / "train").rglob("*.png"))) == 4
    assert len(list((output_dir / "images" / "val").rglob("*.png"))) == 2
    assert len(list((output_dir / "images" / "test").rglob("*.png"))) == 2
    assert len(list((output_dir / "labels" / "train").rglob("*.txt"))) == 4
    assert len(list((output_dir / "labels" / "val").rglob("*.txt"))) == 2
    assert len(list((output_dir / "labels" / "test").rglob("*.txt"))) == 2
    assert (output_dir / "classes.txt").exists()


def test_split_yolo_dataset_endpoint_train_val(client: TestClient, case_dir: Path) -> None:
    dataset_dir = case_dir / "yolo_dataset"
    output_dir = case_dir / "split_out"
    create_yolo_dataset(dataset_dir, sample_count=5)

    response = client.post(
        "/api/v1/preprocess/split-yolo-dataset",
        json={
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "mode": "train_val",
            "train_ratio": 0.6,
            "val_ratio": 0.4,
            "shuffle": False,
            "output_layout": "images_first",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["train_images"] == 3
    assert data["val_images"] == 2
    assert data["test_images"] == 0
    assert len(list((output_dir / "images" / "train").rglob("*.png"))) == 3
    assert len(list((output_dir / "images" / "val").rglob("*.png"))) == 2
    assert not (output_dir / "images" / "test").exists()


def test_split_yolo_dataset_endpoint_train_only(client: TestClient, case_dir: Path) -> None:
    dataset_dir = case_dir / "yolo_dataset"
    output_dir = case_dir / "split_out"
    create_yolo_dataset(dataset_dir, sample_count=4)

    response = client.post(
        "/api/v1/preprocess/split-yolo-dataset",
        json={
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "mode": "train_only",
            "shuffle": False,
            "output_layout": "images_first",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["train_images"] == 4
    assert data["val_images"] == 0
    assert data["test_images"] == 0
    assert len(list((output_dir / "images" / "train").rglob("*.png"))) == 4
    assert not (output_dir / "images" / "val").exists()
    assert not (output_dir / "images" / "test").exists()


def test_split_yolo_dataset_endpoint_split_first_layout(
    client: TestClient, case_dir: Path
) -> None:
    dataset_dir = case_dir / "yolo_dataset"
    output_dir = case_dir / "split_out"
    create_yolo_dataset(dataset_dir, sample_count=6)

    response = client.post(
        "/api/v1/preprocess/split-yolo-dataset",
        json={
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "mode": "train_val_test",
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "shuffle": False,
            "output_layout": "split_first",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["train_images"] == 3
    assert data["val_images"] == 2
    assert data["test_images"] == 1
    assert len(list((output_dir / "train" / "images").rglob("*.png"))) == 3
    assert len(list((output_dir / "val" / "images").rglob("*.png"))) == 2
    assert len(list((output_dir / "test" / "images").rglob("*.png"))) == 1
    assert len(list((output_dir / "train" / "labels").rglob("*.txt"))) == 3
