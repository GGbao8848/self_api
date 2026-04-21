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


def test_class_index_map_explicit_ids_and_training_names(
    client: TestClient, case_dir: Path
) -> None:
    """class_index_map：多源合并到同一 id；training_names 作为 yaml / classes.txt 显示名。"""
    dataset_dir = case_dir / "voc_idx"
    images_dir = dataset_dir / "images"
    xmls_dir = dataset_dir / "xmls"

    for idx, orig in enumerate(["type_a", "type_b", "type_c"], start=1):
        create_image(images_dir / f"i_{idx}.jpg", color=(40, 40, idx * 20), size=(80, 80))
        create_voc_xml(
            xmls_dir / f"i_{idx}.xml",
            filename=f"i_{idx}.jpg",
            size=(80, 80),
            objects=[(orig, (5, 5, 40, 60))],
        )

    response = client.post(
        "/api/v1/preprocess/xml-to-yolo",
        json={
            "input_dir": str(dataset_dir),
            "class_index_map": {"type_a": 0, "type_b": 0, "type_c": 1},
            "training_names": ["merged_ab", "single_c"],
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["classes"] == ["merged_ab", "single_c"]
    assert data["class_to_id"] == {"type_a": 0, "type_b": 0, "type_c": 1}

    for idx in (1, 2):
        t = (dataset_dir / "labels" / f"i_{idx}.txt").read_text(encoding="utf-8").strip()
        assert t.startswith("0 "), f"i_{idx} 应为 id 0"
    t3 = (dataset_dir / "labels" / f"i_3.txt").read_text(encoding="utf-8").strip()
    assert t3.startswith("1 ")

    classes_txt = (dataset_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes_txt == ["merged_ab", "single_c"]


def test_classes_with_training_names_only_changes_display_file(
    client: TestClient, case_dir: Path
) -> None:
    """保留 classes 决定索引；training_names 仅改变 classes.txt 与响应 classes（yaml 用名）。"""
    dataset_dir = case_dir / "voc_disp"
    images_dir = dataset_dir / "images"
    xmls_dir = dataset_dir / "xmls"

    create_image(images_dir / "x.jpg", color=(1, 2, 3), size=(50, 50))
    create_voc_xml(
        xmls_dir / "x.xml",
        filename="x.jpg",
        size=(50, 50),
        objects=[("raw_cat", (5, 5, 30, 40))],
    )

    response = client.post(
        "/api/v1/preprocess/xml-to-yolo",
        json={
            "input_dir": str(dataset_dir),
            "classes": ["raw_cat"],
            "training_names": ["猫"],
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["class_to_id"] == {"raw_cat": 0}
    assert data["classes"] == ["猫"]
    lines = (dataset_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["猫"]
    label = (dataset_dir / "labels" / "x.txt").read_text(encoding="utf-8").strip()
    assert label.startswith("0 ")


def test_class_index_map_non_contiguous_rejected(client: TestClient, case_dir: Path) -> None:
    dataset_dir = case_dir / "voc_bad"
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "xmls").mkdir(parents=True)
    create_image(dataset_dir / "images" / "a.jpg", color=(1, 1, 1), size=(10, 10))
    create_voc_xml(
        dataset_dir / "xmls" / "a.xml",
        filename="a.jpg",
        size=(10, 10),
        objects=[("c", (1, 1, 5, 5))],
    )
    response = client.post(
        "/api/v1/preprocess/xml-to-yolo",
        json={
            "input_dir": str(dataset_dir),
            "class_index_map": {"c": 0, "d": 2},
        },
    )
    assert response.status_code == 400
