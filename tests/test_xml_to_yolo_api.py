from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image, create_voc_xml


def test_xml_to_yolo_endpoint(client: TestClient, case_dir: Path) -> None:
    dataset_dir = case_dir / "voc_dataset"
    images_dir = dataset_dir / "images"
    xmls_dir = dataset_dir / "xmls"

    create_image(images_dir / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    create_image(images_dir / "img_2.jpg", color=(10, 255, 10), size=(100, 100))

    create_voc_xml(
        xmls_dir / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )
    create_voc_xml(
        xmls_dir / "img_2.xml",
        filename="img_2.jpg",
        size=(100, 100),
        objects=[("dog", (20, 30, 70, 90))],
    )

    response = client.post(
        "/api/v1/preprocess/xml-to-yolo",
        json={"input_dir": str(dataset_dir)},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["total_xml_files"] == 2
    assert data["converted_files"] == 2
    assert data["skipped_files"] == 0
    assert data["total_boxes"] == 2
    assert data["classes"] == ["cat", "dog"]
    assert data["class_to_id"] == {"cat": 0, "dog": 1}
    assert (dataset_dir / "classes.txt").exists()

    label_1 = dataset_dir / "labels" / "img_1.txt"
    label_2 = dataset_dir / "labels" / "img_2.txt"
    assert label_1.exists()
    assert label_2.exists()
    assert label_1.read_text(encoding="utf-8").strip().startswith("0 ")
    assert label_2.read_text(encoding="utf-8").strip().startswith("1 ")
