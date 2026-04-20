from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image, create_voc_xml


def test_class_name_map_many_to_one(client: TestClient, case_dir: Path) -> None:
    """louyou1/louyou2/louyou3 合并为 louyou（索引 0）。"""
    dataset_dir = case_dir / "voc_dataset"
    images_dir = dataset_dir / "images"
    xmls_dir = dataset_dir / "xmls"

    for idx, orig_name in enumerate(["louyou1", "louyou2", "louyou3"], start=1):
        create_image(images_dir / f"img_{idx}.jpg", color=(idx * 50, 10, 10), size=(100, 100))
        create_voc_xml(
            xmls_dir / f"img_{idx}.xml",
            filename=f"img_{idx}.jpg",
            size=(100, 100),
            objects=[(orig_name, (10, 10, 60, 60))],
        )

    response = client.post(
        "/api/v1/preprocess/xml-to-yolo",
        json={
            "input_dir": str(dataset_dir),
            "class_name_map": {
                "louyou1": "louyou",
                "louyou2": "louyou",
                "louyou3": "louyou",
            },
            "classes": ["louyou"],
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["classes"] == ["louyou"]
    assert data["class_to_id"] == {"louyou": 0}
    assert data["converted_files"] == 3
    assert data["total_boxes"] == 3

    for idx in (1, 2, 3):
        label = (dataset_dir / "labels" / f"img_{idx}.txt").read_text(encoding="utf-8")
        assert label.strip().startswith("0 "), f"img_{idx} 应被映射到 class_id=0"


def test_class_name_map_partial_merge_keeps_others(
    client: TestClient, case_dir: Path
) -> None:
    """只把 louyou1/louyou2 合并，louyou3 保留原名；final classes 不重复。"""
    dataset_dir = case_dir / "voc_dataset"
    images_dir = dataset_dir / "images"
    xmls_dir = dataset_dir / "xmls"

    for idx, orig_name in enumerate(["louyou1", "louyou2", "louyou3"], start=1):
        create_image(images_dir / f"img_{idx}.jpg", color=(idx * 50, 10, 10), size=(100, 100))
        create_voc_xml(
            xmls_dir / f"img_{idx}.xml",
            filename=f"img_{idx}.jpg",
            size=(100, 100),
            objects=[(orig_name, (10, 10, 60, 60))],
        )

    response = client.post(
        "/api/v1/preprocess/xml-to-yolo",
        json={
            "input_dir": str(dataset_dir),
            "class_name_map": {
                "louyou1": "louyou",
                "louyou2": "louyou",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert set(data["classes"]) == {"louyou", "louyou3"}
    assert data["converted_files"] == 3
    assert data["total_boxes"] == 3

    id_louyou = data["class_to_id"]["louyou"]
    id_louyou3 = data["class_to_id"]["louyou3"]
    assert id_louyou != id_louyou3

    label_1 = (dataset_dir / "labels" / "img_1.txt").read_text(encoding="utf-8").strip()
    label_2 = (dataset_dir / "labels" / "img_2.txt").read_text(encoding="utf-8").strip()
    label_3 = (dataset_dir / "labels" / "img_3.txt").read_text(encoding="utf-8").strip()

    assert label_1.startswith(f"{id_louyou} ")
    assert label_2.startswith(f"{id_louyou} ")
    assert label_3.startswith(f"{id_louyou3} ")
