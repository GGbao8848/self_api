from pathlib import Path

from fastapi.testclient import TestClient


def test_scan_yolo_label_indices_recursive(client: TestClient, case_dir: Path) -> None:
    root = case_dir / "dataset"
    labels_dir = root / "labels"
    nested_labels_dir = labels_dir / "nested"
    nested_labels_dir.mkdir(parents=True)

    (labels_dir / "a.txt").write_text(
        "0 0.1 0.2 0.3 0.4\n1 0.2 0.3 0.4 0.5\n3 0.3 0.4 0.5 0.6\n",
        encoding="utf-8",
    )
    (nested_labels_dir / "b.txt").write_text(
        "1 0.1 0.2 0.3 0.4\n18 0.2 0.3 0.4 0.5\nbad line\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/scan-yolo-label-indices",
        json={"input_dir": str(root)},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["input_dir"] == str(root)
    assert data["labels_dir"] == str(labels_dir)
    assert data["total_label_files"] == 2
    assert data["total_objects"] == 5
    assert data["skipped_invalid_lines"] == 1
    assert data["indices"] == [
        {"index": 0, "count": 1},
        {"index": 1, "count": 2},
        {"index": 3, "count": 1},
        {"index": 18, "count": 1},
    ]


def test_rewrite_yolo_label_indices_with_mapping_and_default(
    client: TestClient,
    case_dir: Path,
) -> None:
    root = case_dir / "dataset"
    labels_dir = root / "labels"
    nested_labels_dir = labels_dir / "nested"
    nested_labels_dir.mkdir(parents=True)

    (labels_dir / "a.txt").write_text(
        "0 0.1 0.2 0.3 0.4\n5 0.2 0.3 0.4 0.5\n8 0.3 0.4 0.5 0.6\n",
        encoding="utf-8",
    )
    (nested_labels_dir / "b.txt").write_text(
        "2 0.1 0.2 0.3 0.4\n18 0.2 0.3 0.4 0.5\nbad line\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/rewrite-yolo-label-indices",
        json={
            "input_dir": str(root),
            "mapping": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
            "default_target_index": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_label_files"] == 2
    assert data["total_objects"] == 5
    assert data["modified_label_files"] == 2
    assert data["unchanged_label_files"] == 0
    assert data["changed_lines"] == 3
    assert data["skipped_invalid_lines"] == 1
    assert data["mapping"] == {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    assert data["default_target_index"] == 1
    assert (labels_dir / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "0 0.1 0.2 0.3 0.4",
        "0 0.2 0.3 0.4 0.5",
        "1 0.3 0.4 0.5 0.6",
    ]
    assert (nested_labels_dir / "b.txt").read_text(encoding="utf-8").splitlines() == [
        "0 0.1 0.2 0.3 0.4",
        "1 0.2 0.3 0.4 0.5",
        "bad line",
    ]


def test_rewrite_yolo_label_indices_set_all_to_one(client: TestClient, case_dir: Path) -> None:
    root = case_dir / "dataset"
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text(
        "0 0.1 0.2 0.3 0.4\n2 0.2 0.3 0.4 0.5\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/preprocess/rewrite-yolo-label-indices",
        json={"input_dir": str(root), "default_target_index": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["changed_lines"] == 2
    assert (labels_dir / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "1 0.1 0.2 0.3 0.4",
        "1 0.2 0.3 0.4 0.5",
    ]


def test_rewrite_yolo_label_indices_requires_rule(client: TestClient, case_dir: Path) -> None:
    root = case_dir / "dataset"
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/rewrite-yolo-label-indices",
        json={"input_dir": str(root)},
    )

    assert response.status_code == 422


def test_scan_yolo_label_indices_accepts_labels_dir_directly(
    client: TestClient,
    case_dir: Path,
) -> None:
    labels_dir = case_dir / "dataset" / "labels"
    nested_labels_dir = labels_dir / "nested"
    nested_labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    (nested_labels_dir / "b.txt").write_text("2 0.2 0.3 0.4 0.5\n", encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/scan-yolo-label-indices",
        json={"input_dir": str(labels_dir)},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["input_dir"] == str(labels_dir)
    assert data["labels_dir"] == str(labels_dir)
    assert data["total_label_files"] == 2
    assert data["total_objects"] == 2
    assert data["indices"] == [
        {"index": 0, "count": 1},
        {"index": 2, "count": 1},
    ]


def test_rewrite_yolo_label_indices_accepts_labels_dir_directly(
    client: TestClient,
    case_dir: Path,
) -> None:
    labels_dir = case_dir / "dataset" / "labels"
    nested_labels_dir = labels_dir / "nested"
    nested_labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    (nested_labels_dir / "b.txt").write_text("2 0.2 0.3 0.4 0.5\n", encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/rewrite-yolo-label-indices",
        json={"input_dir": str(labels_dir), "default_target_index": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["input_dir"] == str(labels_dir)
    assert data["labels_dir"] == str(labels_dir)
    assert data["changed_lines"] == 2
    assert (labels_dir / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "1 0.1 0.2 0.3 0.4",
    ]
    assert (nested_labels_dir / "b.txt").read_text(encoding="utf-8").splitlines() == [
        "1 0.2 0.3 0.4 0.5",
    ]


def test_scan_yolo_label_indices_discovers_nested_labels_dirs(
    client: TestClient,
    case_dir: Path,
) -> None:
    root = case_dir / "outer"
    labels_dir_a = root / "part_a" / "labels"
    labels_dir_b = root / "part_b" / "sub" / "labels"
    labels_dir_a.mkdir(parents=True)
    labels_dir_b.mkdir(parents=True)
    (labels_dir_a / "a.txt").write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    (labels_dir_b / "b.txt").write_text("3 0.2 0.3 0.4 0.5\n", encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/scan-yolo-label-indices",
        json={"input_dir": str(root)},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["input_dir"] == str(root)
    assert data["total_label_files"] == 2
    assert sorted(data["labels_dirs"]) == sorted([str(labels_dir_a), str(labels_dir_b)])
    assert data["indices"] == [
        {"index": 0, "count": 1},
        {"index": 3, "count": 1},
    ]


def test_rewrite_yolo_label_indices_discovers_nested_labels_dirs(
    client: TestClient,
    case_dir: Path,
) -> None:
    root = case_dir / "outer"
    labels_dir_a = root / "part_a" / "labels"
    labels_dir_b = root / "part_b" / "sub" / "labels"
    labels_dir_a.mkdir(parents=True)
    labels_dir_b.mkdir(parents=True)
    (labels_dir_a / "a.txt").write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    (labels_dir_b / "b.txt").write_text("3 0.2 0.3 0.4 0.5\n", encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/rewrite-yolo-label-indices",
        json={"input_dir": str(root), "default_target_index": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["input_dir"] == str(root)
    assert data["changed_lines"] == 2
    assert sorted(data["labels_dirs"]) == sorted([str(labels_dir_a), str(labels_dir_b)])
    assert (labels_dir_a / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "1 0.1 0.2 0.3 0.4",
    ]
    assert (labels_dir_b / "b.txt").read_text(encoding="utf-8").splitlines() == [
        "1 0.2 0.3 0.4 0.5",
    ]
