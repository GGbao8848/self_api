from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _create_image(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    image = Image.new("RGB", size=size, color=color)
    image.save(path)


def test_healthz() -> None:
    response = client.get("/api/v1/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sliding_window_crop_endpoint(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    _create_image(input_dir / "sample.png", color=(255, 0, 0), size=(10, 10))

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "window_width": 4,
        "window_height": 4,
        "stride_x": 4,
        "stride_y": 4,
        "include_partial_edges": False,
        "recursive": True,
        "keep_subdirs": True,
        "output_format": "png",
    }

    response = client.post("/api/v1/preprocess/sliding-window-crop", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["generated_crops"] == 4
    assert data["processed_images"] == 1
    assert len(list(output_dir.rglob("*.png"))) == 4


def test_deduplicate_endpoint_md5(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    unique_dir = tmp_path / "unique"
    report_path = tmp_path / "reports" / "dedup.json"
    input_dir.mkdir(parents=True, exist_ok=True)

    image1 = input_dir / "a.png"
    image2 = input_dir / "b.png"
    image3 = input_dir / "c.png"

    _create_image(image1, color=(10, 10, 10), size=(8, 8))
    image2.write_bytes(image1.read_bytes())
    _create_image(image3, color=(200, 200, 200), size=(8, 8))

    payload = {
        "input_dir": str(input_dir),
        "method": "md5",
        "copy_unique_to": str(unique_dir),
        "report_path": str(report_path),
    }

    response = client.post("/api/v1/preprocess/deduplicate", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["total_images"] == 3
    assert data["unique_images"] == 2
    assert data["duplicate_images"] == 1
    assert len(data["groups"]) == 1
    assert report_path.exists()
    assert len(list(unique_dir.rglob("*.png"))) == 2
