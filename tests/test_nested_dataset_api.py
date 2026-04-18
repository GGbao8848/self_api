import time
from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image, create_voc_xml


def _wait_task_done(
    client: TestClient,
    task_id: str,
    *,
    timeout_seconds: float = 8.0,
) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/api/v1/preprocess/tasks/{task_id}")
        assert response.status_code == 200
        task_data = response.json()
        if task_data["state"] in {"succeeded", "failed"}:
            return task_data
        time.sleep(0.05)
    raise AssertionError(f"task did not complete in {timeout_seconds} seconds: {task_id}")


def test_discover_leaf_dirs_endpoint(client: TestClient, case_dir: Path) -> None:
    root_dir = case_dir / "nested_root"
    create_image(root_dir / "group_a" / "leaf_1" / "a.jpg", color=(255, 0, 0), size=(32, 32))
    create_voc_xml(
        root_dir / "group_a" / "leaf_1" / "a.xml",
        filename="a.jpg",
        size=(32, 32),
        objects=[("cat", (2, 2, 10, 10))],
    )
    create_image(root_dir / "group_b" / "b.jpg", color=(0, 255, 0), size=(32, 32))
    create_image(root_dir / "group_c" / "parent.jpg", color=(0, 0, 255), size=(32, 32))
    create_image(root_dir / "group_c" / "deep" / "c.jpg", color=(0, 0, 128), size=(32, 32))

    response = client.post(
        "/api/v1/preprocess/discover-leaf-dirs",
        json={"input_dir": str(root_dir)},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["total_leaf_dirs"] == 3
    assert data["leaf_dirs"] == [
        str(root_dir / "group_a" / "leaf_1"),
        str(root_dir / "group_b"),
        str(root_dir / "group_c" / "deep"),
    ]


def test_clean_nested_dataset_endpoint(client: TestClient, case_dir: Path) -> None:
    root_dir = case_dir / "raw_nested"
    leaf_dir = root_dir / "project" / "batch_1"
    create_image(leaf_dir / "has_label.jpg", color=(255, 0, 0), size=(40, 40))
    create_image(leaf_dir / "no_label.jpg", color=(0, 255, 0), size=(40, 40))
    create_image(leaf_dir / "empty_label.jpg", color=(0, 0, 255), size=(40, 40))
    create_voc_xml(
        leaf_dir / "has_label.xml",
        filename="has_label.jpg",
        size=(40, 40),
        objects=[("cat", (5, 5, 20, 20))],
    )
    create_voc_xml(
        leaf_dir / "empty_label.xml",
        filename="empty_label.jpg",
        size=(40, 40),
        objects=[],
    )
    create_voc_xml(
        leaf_dir / "orphan.xml",
        filename="missing.jpg",
        size=(40, 40),
        objects=[("cat", (5, 5, 20, 20))],
    )

    output_dir = case_dir / "cleaned"
    response = client.post(
        "/api/v1/preprocess/clean-nested-dataset",
        json={
            "input_dir": str(root_dir),
            "output_dir": str(output_dir),
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["discovered_leaf_dirs"] == 1
    assert data["processed_leaf_dirs"] == 1
    assert data["total_images"] == 3
    assert data["labeled_images"] == 1
    assert data["background_images"] == 2
    assert data["copied_xml_files"] == 1
    assert data["empty_or_invalid_xml_files"] == 1
    assert data["orphan_xml_files"] == 1

    cleaned_leaf = output_dir / "project" / "batch_1"
    assert (cleaned_leaf / "images" / "has_label.jpg").exists()
    assert (cleaned_leaf / "xmls" / "has_label.xml").exists()
    assert (cleaned_leaf / "backgrounds" / "no_label.jpg").exists()
    assert (cleaned_leaf / "backgrounds" / "empty_label.jpg").exists()


def test_clean_nested_dataset_flatten_endpoint(client: TestClient, case_dir: Path) -> None:
    root_dir = case_dir / "flatten_nested"
    leaf_a = root_dir / "branch_a" / "leaf"
    leaf_b = root_dir / "branch_b" / "leaf"
    create_image(leaf_a / "same.jpg", color=(255, 0, 0), size=(8, 8))
    create_voc_xml(
        leaf_a / "same.xml",
        filename="same.jpg",
        size=(8, 8),
        objects=[("cat", (1, 1, 4, 4))],
    )
    create_image(leaf_b / "same.jpg", color=(0, 255, 0), size=(8, 8))
    create_voc_xml(
        leaf_b / "same.xml",
        filename="same.jpg",
        size=(8, 8),
        objects=[("dog", (1, 1, 4, 4))],
    )

    output_dir = case_dir / "flatten_out"
    response = client.post(
        "/api/v1/preprocess/clean-nested-dataset",
        json={
            "input_dir": str(root_dir),
            "output_dir": str(output_dir),
            "flatten": True,
        },
    )
    data = response.json()
    assert response.status_code == 200
    assert data["discovered_leaf_dirs"] == 2
    assert data["labeled_images"] == 2
    images_dir = output_dir / "images"
    xmls_dir = output_dir / "xmls"
    assert images_dir.is_dir()
    assert xmls_dir.is_dir()
    names = {p.name for p in images_dir.glob("*.jpg")}
    assert names == {
        "branch_a__leaf__same.jpg",
        "branch_b__leaf__same.jpg",
    }
    assert {p.name for p in xmls_dir.glob("*.xml")} == {
        "branch_a__leaf__same.xml",
        "branch_b__leaf__same.xml",
    }


def test_clean_nested_dataset_images_xmls_subfolders_endpoint(client: TestClient, case_dir: Path) -> None:
    """VOC-style: sample dir contains images/ and xmls/ subfolders."""
    root_dir = case_dir / "voc_style"
    sample = root_dir / "session_a"
    create_image(sample / "images" / "a.jpg", color=(255, 0, 0), size=(16, 16))
    create_image(sample / "images" / "bg.jpg", color=(0, 255, 0), size=(16, 16))
    create_voc_xml(
        sample / "xmls" / "a.xml",
        filename="a.jpg",
        size=(16, 16),
        objects=[("cat", (2, 2, 8, 8))],
    )

    output_dir = case_dir / "voc_cleaned"
    response = client.post(
        "/api/v1/preprocess/clean-nested-dataset",
        json={
            "input_dir": str(root_dir),
            "output_dir": str(output_dir),
            "pairing_mode": "images_xmls_subfolders",
        },
    )
    data = response.json()
    assert response.status_code == 200
    assert data["discovered_leaf_dirs"] == 1
    assert data["labeled_images"] == 1
    assert data["background_images"] == 1
    assert data["copied_xml_files"] == 1

    out_leaf = output_dir / "session_a"
    assert (out_leaf / "images" / "a.jpg").exists()
    assert (out_leaf / "xmls" / "a.xml").exists()
    assert (out_leaf / "backgrounds" / "bg.jpg").exists()


def test_clean_nested_dataset_auto_detects_images_xmls_subfolders(
    client: TestClient,
    case_dir: Path,
) -> None:
    root_dir = case_dir / "voc_style_auto"
    sample = root_dir / "session_auto"
    create_image(sample / "images" / "a.jpg", color=(255, 0, 0), size=(16, 16))
    create_image(sample / "images" / "bg.jpg", color=(0, 255, 0), size=(16, 16))
    create_voc_xml(
        sample / "xmls" / "a.xml",
        filename="a.jpg",
        size=(16, 16),
        objects=[("cat", (2, 2, 8, 8))],
    )

    output_dir = case_dir / "voc_cleaned_auto"
    response = client.post(
        "/api/v1/preprocess/clean-nested-dataset",
        json={
            "input_dir": str(root_dir),
            "output_dir": str(output_dir),
        },
    )
    data = response.json()
    assert response.status_code == 200
    assert data["discovered_leaf_dirs"] == 1
    assert data["labeled_images"] == 1
    assert data["background_images"] == 1
    assert data["copied_xml_files"] == 1

    out_leaf = output_dir / "session_auto"
    assert (out_leaf / "images" / "a.jpg").exists()
    assert (out_leaf / "xmls" / "a.xml").exists()
    assert (out_leaf / "backgrounds" / "bg.jpg").exists()


def test_clean_nested_dataset_skip_backgrounds_endpoint(client: TestClient, case_dir: Path) -> None:
    root_dir = case_dir / "raw_nested_skip_bg"
    leaf_dir = root_dir / "project" / "batch_1"
    create_image(leaf_dir / "has_label.jpg", color=(255, 0, 0), size=(40, 40))
    create_image(leaf_dir / "no_label.jpg", color=(0, 255, 0), size=(40, 40))
    create_voc_xml(
        leaf_dir / "has_label.xml",
        filename="has_label.jpg",
        size=(40, 40),
        objects=[("cat", (5, 5, 20, 20))],
    )

    output_dir = case_dir / "cleaned_skip_bg"
    response = client.post(
        "/api/v1/preprocess/clean-nested-dataset",
        json={
            "input_dir": str(root_dir),
            "output_dir": str(output_dir),
            "include_backgrounds": False,
        },
    )
    data = response.json()
    assert response.status_code == 200
    assert data["labeled_images"] == 1
    assert data["background_images"] == 0
    assert data["skipped_unlabeled_images"] == 1

    cleaned_leaf = output_dir / "project" / "batch_1"
    assert (cleaned_leaf / "images" / "has_label.jpg").exists()
    assert (cleaned_leaf / "xmls" / "has_label.xml").exists()
    assert not (cleaned_leaf / "backgrounds").exists()


def test_aggregate_nested_dataset_endpoint(client: TestClient, case_dir: Path) -> None:
    cleaned_root = case_dir / "cleaned_nested"

    fragment_a = cleaned_root / "group_a" / "leaf_a"
    create_image(fragment_a / "images" / "same_name.jpg", color=(255, 0, 0), size=(24, 24))
    (fragment_a / "labels" / "same_name.txt").parent.mkdir(parents=True, exist_ok=True)
    (fragment_a / "labels" / "same_name.txt").write_text(
        "0 0.500000 0.500000 0.400000 0.400000\n",
        encoding="utf-8",
    )
    (fragment_a / "classes.txt").write_text("dog\ncat\n", encoding="utf-8")
    create_image(fragment_a / "backgrounds" / "bg.jpg", color=(128, 0, 0), size=(24, 24))

    fragment_b = cleaned_root / "group_b" / "leaf_b"
    create_image(fragment_b / "images" / "same_name.jpg", color=(0, 255, 0), size=(24, 24))
    (fragment_b / "labels" / "same_name.txt").parent.mkdir(parents=True, exist_ok=True)
    (fragment_b / "labels" / "same_name.txt").write_text(
        "0 0.250000 0.250000 0.300000 0.300000\n",
        encoding="utf-8",
    )
    (fragment_b / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")

    output_dir = case_dir / "dataset"
    response = client.post(
        "/api/v1/preprocess/aggregate-nested-dataset",
        json={
            "input_dir": str(cleaned_root),
            "output_dir": str(output_dir),
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["fragment_dirs"] == 2
    assert data["aggregated_images"] == 2
    assert data["aggregated_backgrounds"] == 1
    assert data["skipped_images"] == 0
    assert data["classes"] == ["dog", "cat"]
    assert data["class_to_id"] == {"dog": 0, "cat": 1}
    assert (output_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == ["dog", "cat"]

    image_files = sorted((output_dir / "images").glob("*.jpg"))
    label_files = sorted((output_dir / "labels").glob("*.txt"))
    background_files = sorted((output_dir / "backgrounds").glob("*.jpg"))
    assert len(image_files) == 2
    assert len(label_files) == 2
    assert len(background_files) == 1
    assert len({path.name for path in image_files}) == 2
    assert any(path.read_text(encoding="utf-8").startswith("1 ") for path in label_files)
    assert (output_dir / "manifest.json").exists()


def test_clean_nested_dataset_async_endpoint(
    client: TestClient,
    case_dir: Path,
    isolated_runtime,
) -> None:
    root_dir = case_dir / "async_nested"
    create_image(root_dir / "leaf" / "one.jpg", color=(255, 0, 0), size=(20, 20))
    create_voc_xml(
        root_dir / "leaf" / "one.xml",
        filename="one.jpg",
        size=(20, 20),
        objects=[("cat", (2, 2, 10, 10))],
    )

    output_dir = case_dir / "async_cleaned"
    submit_resp = client.post(
        "/api/v1/preprocess/clean-nested-dataset/async",
        json={
            "input_dir": str(root_dir),
            "output_dir": str(output_dir),
        },
    )
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()
    assert submit_data["task_type"] == "clean_nested_dataset"

    task_data = _wait_task_done(client, submit_data["task_id"])
    assert task_data["state"] == "succeeded"
    assert task_data["result"]["labeled_images"] == 1
    assert (output_dir / "leaf" / "images" / "one.jpg").exists()
