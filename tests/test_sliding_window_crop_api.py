from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image


def test_sliding_window_crop_endpoint(client: TestClient, case_dir: Path) -> None:
    input_dir = case_dir / "input"
    output_dir = case_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    create_image(input_dir / "sample.png", color=(255, 0, 0), size=(10, 10))

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
