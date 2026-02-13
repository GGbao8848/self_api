from pathlib import Path
from xml.etree import ElementTree

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _create_image(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    image = Image.new("RGB", size=size, color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _create_voc_xml(
    path: Path,
    filename: str,
    size: tuple[int, int],
    objects: list[tuple[str, tuple[int, int, int, int]]],
) -> None:
    root = ElementTree.Element("annotation")
    filename_node = ElementTree.SubElement(root, "filename")
    filename_node.text = filename

    size_node = ElementTree.SubElement(root, "size")
    width_node = ElementTree.SubElement(size_node, "width")
    width_node.text = str(size[0])
    height_node = ElementTree.SubElement(size_node, "height")
    height_node.text = str(size[1])
    depth_node = ElementTree.SubElement(size_node, "depth")
    depth_node.text = "3"

    for class_name, (xmin, ymin, xmax, ymax) in objects:
        object_node = ElementTree.SubElement(root, "object")
        name_node = ElementTree.SubElement(object_node, "name")
        name_node.text = class_name
        difficult_node = ElementTree.SubElement(object_node, "difficult")
        difficult_node.text = "0"
        bbox_node = ElementTree.SubElement(object_node, "bndbox")
        for tag, value in (
            ("xmin", xmin),
            ("ymin", ymin),
            ("xmax", xmax),
            ("ymax", ymax),
        ):
            coord = ElementTree.SubElement(bbox_node, tag)
            coord.text = str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _create_yolo_dataset(root: Path, sample_count: int) -> None:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(sample_count):
        image_name = f"sample_{idx:02d}.png"
        label_name = f"sample_{idx:02d}.txt"
        _create_image(images_dir / image_name, color=(idx, idx, idx), size=(100, 100))
        (labels_dir / label_name).write_text("0 0.500000 0.500000 0.400000 0.400000\n", encoding="utf-8")

    (root / "classes.txt").write_text("object\n", encoding="utf-8")


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


def test_xml_to_yolo_endpoint(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "voc_dataset"
    images_dir = dataset_dir / "images"
    xmls_dir = dataset_dir / "xmls"

    _create_image(images_dir / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    _create_image(images_dir / "img_2.jpg", color=(10, 255, 10), size=(100, 100))

    _create_voc_xml(
        xmls_dir / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )
    _create_voc_xml(
        xmls_dir / "img_2.xml",
        filename="img_2.jpg",
        size=(100, 100),
        objects=[("dog", (20, 30, 70, 90))],
    )

    response = client.post(
        "/api/v1/preprocess/xml-to-yolo",
        json={"dataset_dir": str(dataset_dir)},
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


def test_split_yolo_dataset_endpoint_train_val_test(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "yolo_dataset"
    output_dir = tmp_path / "split_out"
    _create_yolo_dataset(dataset_dir, sample_count=8)

    response = client.post(
        "/api/v1/preprocess/split-yolo-dataset",
        json={
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "mode": "train_val_test",
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "shuffle": False,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["total_images"] == 8
    assert data["paired_images"] == 8
    assert data["skipped_images"] == 0
    assert data["train_images"] == 4
    assert data["val_images"] == 2
    assert data["test_images"] == 2
    assert len(list((output_dir / "images" / "train").rglob("*.png"))) == 4
    assert len(list((output_dir / "images" / "val").rglob("*.png"))) == 2
    assert len(list((output_dir / "images" / "test").rglob("*.png"))) == 2
    assert len(list((output_dir / "labels" / "train").rglob("*.txt"))) == 4
    assert len(list((output_dir / "labels" / "val").rglob("*.txt"))) == 2
    assert len(list((output_dir / "labels" / "test").rglob("*.txt"))) == 2
    assert (output_dir / "classes.txt").exists()


def test_split_yolo_dataset_endpoint_train_val(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "yolo_dataset"
    output_dir = tmp_path / "split_out"
    _create_yolo_dataset(dataset_dir, sample_count=5)

    response = client.post(
        "/api/v1/preprocess/split-yolo-dataset",
        json={
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "mode": "train_val",
            "train_ratio": 0.6,
            "val_ratio": 0.4,
            "shuffle": False,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["train_images"] == 3
    assert data["val_images"] == 2
    assert data["test_images"] == 0
    assert len(list((output_dir / "images" / "train").rglob("*.png"))) == 3
    assert len(list((output_dir / "images" / "val").rglob("*.png"))) == 2
    assert not (output_dir / "images" / "test").exists()


def test_split_yolo_dataset_endpoint_train_only(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "yolo_dataset"
    output_dir = tmp_path / "split_out"
    _create_yolo_dataset(dataset_dir, sample_count=4)

    response = client.post(
        "/api/v1/preprocess/split-yolo-dataset",
        json={
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "mode": "train_only",
            "shuffle": False,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["train_images"] == 4
    assert data["val_images"] == 0
    assert data["test_images"] == 0
    assert len(list((output_dir / "images" / "train").rglob("*.png"))) == 4
    assert not (output_dir / "images" / "val").exists()
    assert not (output_dir / "images" / "test").exists()
