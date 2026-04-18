"""Tests for YOLO square sliding window crop API."""

from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image


def test_yolo_sliding_window_crop_wide_image(
    client: TestClient,
    case_dir: Path,
) -> None:
    """Wide image (W>H): square window=H, horizontal sliding."""
    images_dir = case_dir / "images"
    labels_dir = case_dir / "labels"
    output_dir = case_dir / "data_crops"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 16x8 wide image: win=8, step=2 (stride_ratio=0.25)
    create_image(images_dir / "wide.png", color=(200, 10, 10), size=(16, 8))
    (labels_dir / "wide.txt").write_text(
        "0 0.250000 0.250000 0.250000 0.250000\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/yolo-sliding-window-crop",
        json={
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "output_dir": str(output_dir),
            "min_vis_ratio": 0.5,
            "stride_ratio": 0.25,
            "ignore_vis_ratio": 0.05,
            "only_wide": True,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["input_images"] == 1
    assert data["processed_images"] == 1
    assert data["generated_crops"] >= 1
    assert data["generated_labels"] >= 1

    out_images = list((output_dir / "images").rglob("*.png"))
    out_labels = list((output_dir / "labels").rglob("*.txt"))
    assert len(out_images) >= 1
    assert len(out_labels) >= 1


def test_yolo_sliding_window_crop_only_wide_skips_square(
    client: TestClient,
    case_dir: Path,
) -> None:
    """Square image (W=H) with only_wide=True: skipped."""
    images_dir = case_dir / "images"
    labels_dir = case_dir / "labels"
    output_dir = case_dir / "data_crops"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    create_image(images_dir / "square.png", color=(10, 200, 10), size=(8, 8))
    (labels_dir / "square.txt").write_text(
        "0 0.500000 0.500000 0.400000 0.400000\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/yolo-sliding-window-crop",
        json={
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "output_dir": str(output_dir),
            "only_wide": True,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["input_images"] == 1
    assert data["processed_images"] == 0
    assert data["skipped_images"] == 1
    assert data["generated_crops"] == 0


def test_yolo_sliding_window_crop_only_wide_false_processes_square(
    client: TestClient,
    case_dir: Path,
) -> None:
    """Square image with only_wide=False: one window."""
    images_dir = case_dir / "images"
    labels_dir = case_dir / "labels"
    output_dir = case_dir / "data_crops"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    create_image(images_dir / "square.png", color=(10, 200, 10), size=(8, 8))
    (labels_dir / "square.txt").write_text(
        "0 0.500000 0.500000 0.400000 0.400000\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/yolo-sliding-window-crop",
        json={
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "output_dir": str(output_dir),
            "only_wide": False,
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
    assert out_labels[0].read_text(encoding="utf-8").strip().startswith("0 ")


def test_yolo_sliding_window_crop_supports_manual_window_and_stride(
    client: TestClient,
    case_dir: Path,
) -> None:
    images_dir = case_dir / "images"
    labels_dir = case_dir / "labels"
    output_dir = case_dir / "data_crops"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    create_image(images_dir / "grid.png", color=(10, 10, 200), size=(10, 10))
    (labels_dir / "grid.txt").write_text(
        "0 0.200000 0.200000 0.200000 0.200000\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/yolo-sliding-window-crop",
        json={
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "output_dir": str(output_dir),
            "window_width": 4,
            "window_height": 4,
            "stride_x": 3,
            "stride_y": 3,
            "only_wide": False,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["processed_images"] == 1
    assert data["generated_crops"] == 1
    assert data["generated_labels"] == 1

    out_images = list((output_dir / "images").rglob("*.png"))
    out_labels = list((output_dir / "labels").rglob("*.txt"))
    assert len(out_images) == 1
    assert len(out_labels) == 1
    assert "_w4_h4_" in out_images[0].name


def test_yolo_sliding_window_crop_without_labels_only_writes_images(
    client: TestClient,
    case_dir: Path,
) -> None:
    images_dir = case_dir / "images"
    output_dir = case_dir / "data_crops"

    images_dir.mkdir(parents=True, exist_ok=True)

    create_image(images_dir / "plain.png", color=(50, 50, 50), size=(10, 10))

    response = client.post(
        "/api/v1/preprocess/yolo-sliding-window-crop",
        json={
            "images_dir": str(images_dir),
            "output_dir": str(output_dir),
            "window_width": 4,
            "window_height": 4,
            "stride_x": 4,
            "stride_y": 4,
            "only_wide": False,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["labels_dir"] is None
    assert data["processed_images"] == 1
    assert data["generated_crops"] == 9
    assert data["generated_labels"] == 0

    out_images = list((output_dir / "images").rglob("*.png"))
    out_labels_dir = output_dir / "labels"
    assert len(out_images) == 9
    assert not out_labels_dir.exists()
