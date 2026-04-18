from pathlib import Path

from fastapi.testclient import TestClient


def test_reset_yolo_label_index_minimal(client: TestClient, case_dir: Path) -> None:
    root = case_dir / "dataset"
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text("3 0.1 0.2 0.3 0.4\n0 0.2 0.3 0.4 0.5\n", encoding="utf-8")
    (labels_dir / "b.txt").write_text("bad line\n", encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/reset-yolo-label-index",
        json={"input_dir": str(root)},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["input_dir"] == str(root)
    assert data["labels_dir"] == str(labels_dir)
    assert data["total_label_files"] == 2
    assert data["modified_label_files"] == 1
    assert data["unchanged_label_files"] == 1
    assert data["changed_lines"] == 1
    assert data["skipped_invalid_lines"] == 1
    assert (labels_dir / "a.txt").read_text(encoding="utf-8").startswith("0 ")


def test_reset_yolo_label_index_requires_labels_dir(client: TestClient, case_dir: Path) -> None:
    root = case_dir / "dataset_missing_labels"
    root.mkdir(parents=True)

    response = client.post(
        "/api/v1/preprocess/reset-yolo-label-index",
        json={"input_dir": str(root)},
    )

    assert response.status_code == 400
