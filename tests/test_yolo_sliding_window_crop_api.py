from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image


def test_yolo_sliding_window_crop_keep_only_labeled_windows(
    client: TestClient,
    case_dir: Path,
) -> None:
    dataset_dir = case_dir / "yolo_large"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    output_dir = case_dir / "yolo_small"

    create_image(images_dir / "big.png", color=(200, 10, 10), size=(8, 8))
    (labels_dir / "big.txt").parent.mkdir(parents=True, exist_ok=True)
    (labels_dir / "big.txt").write_text(
        "0 0.250000 0.250000 0.250000 0.250000\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/yolo-sliding-window-crop",
        json={
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "window_width": 4,
            "window_height": 4,
            "stride_x": 4,
            "stride_y": 4,
            "keep_empty_labels": False,
            "include_partial_edges": False,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["input_images"] == 1
    assert data["processed_images"] == 1
    assert data["generated_crops"] == 1
    assert data["generated_labels"] == 1

    out_images = list((output_dir / "images").rglob("*.png"))
    out_labels = list((output_dir / "labels").rglob("*.txt"))
    assert len(out_images) == 1
    assert len(out_labels) == 1
    assert out_labels[0].read_text(encoding="utf-8").strip() == "0 0.500000 0.500000 0.500000 0.500000"


def test_yolo_sliding_window_crop_keep_empty_windows(
    client: TestClient,
    case_dir: Path,
) -> None:
    dataset_dir = case_dir / "yolo_large"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    output_dir = case_dir / "yolo_small"

    create_image(images_dir / "big.png", color=(10, 200, 10), size=(8, 8))
    (labels_dir / "big.txt").parent.mkdir(parents=True, exist_ok=True)
    (labels_dir / "big.txt").write_text(
        "0 0.250000 0.250000 0.250000 0.250000\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/yolo-sliding-window-crop",
        json={
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "window_width": 4,
            "window_height": 4,
            "stride_x": 4,
            "stride_y": 4,
            "keep_empty_labels": True,
            "include_partial_edges": False,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["generated_crops"] == 4
    assert data["generated_labels"] == 1

    out_images = list((output_dir / "images").rglob("*.png"))
    out_labels = list((output_dir / "labels").rglob("*.txt"))
    assert len(out_images) == 4
    assert len(out_labels) == 4
    non_empty_label_files = [
        path for path in out_labels if path.read_text(encoding="utf-8").strip()
    ]
    assert len(non_empty_label_files) == 1
