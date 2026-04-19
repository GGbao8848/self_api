"""Tests for YOLO augment API."""

from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image


def test_yolo_augment_basic_root_layout(
    client: TestClient,
    case_dir: Path,
) -> None:
    images_dir = case_dir / "images"
    labels_dir = case_dir / "labels"
    output_dir = case_dir / "augment_out"

    create_image(images_dir / "plain.png", color=(10, 20, 30), size=(12, 12))
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "plain.txt").write_text(
        "0 0.500000 0.500000 0.400000 0.400000\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/yolo-augment",
        json={
            "input_dir": str(case_dir),
            "output_dir": str(output_dir),
            "vertical_flip": False,
            "brightness_up": False,
            "brightness_down": False,
            "contrast_up": False,
            "contrast_down": False,
            "gaussian_blur": False,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["processed_images"] == 1
    assert data["generated_images"] == 1
    assert data["generated_labels"] == 1
    assert len(list((output_dir / "images").rglob("*.png"))) == 1
    assert len(list((output_dir / "labels").rglob("*.txt"))) == 1


def test_yolo_augment_supports_split_dataset_tree(
    client: TestClient,
    case_dir: Path,
) -> None:
    input_dir = case_dir / "split_dataset"
    output_dir = case_dir / "split_dataset_augment"

    train_images = input_dir / "train" / "images"
    train_labels = input_dir / "train" / "labels"
    val_images = input_dir / "val" / "images"
    val_labels = input_dir / "val" / "labels"

    create_image(train_images / "train_a.png", color=(20, 20, 200), size=(8, 8))
    train_labels.mkdir(parents=True, exist_ok=True)
    (train_labels / "train_a.txt").write_text(
        "0 0.250000 0.500000 0.200000 0.200000\n",
        encoding="utf-8",
    )

    create_image(val_images / "val_a.png", color=(200, 20, 20), size=(8, 8))
    val_labels.mkdir(parents=True, exist_ok=True)
    (val_labels / "val_a.txt").write_text(
        "0 0.750000 0.500000 0.200000 0.200000\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/yolo-augment",
        json={
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "vertical_flip": False,
            "brightness_up": False,
            "brightness_down": False,
            "contrast_up": False,
            "contrast_down": False,
            "gaussian_blur": False,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["processed_images"] == 2
    assert data["generated_images"] == 2
    assert data["generated_labels"] == 2

    train_out_images = list((output_dir / "train" / "images").rglob("*.png"))
    train_out_labels = list((output_dir / "train" / "labels").rglob("*.txt"))
    val_out_images = list((output_dir / "val" / "images").rglob("*.png"))
    val_out_labels = list((output_dir / "val" / "labels").rglob("*.txt"))

    assert len(train_out_images) == 1
    assert len(train_out_labels) == 1
    assert len(val_out_images) == 1
    assert len(val_out_labels) == 1
    assert train_out_labels[0].read_text(encoding="utf-8").strip().startswith("0 ")
