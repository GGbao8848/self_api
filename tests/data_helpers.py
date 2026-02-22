from pathlib import Path
from xml.etree import ElementTree

from PIL import Image


def create_image(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size=size, color=color)
    image.save(path)


def create_voc_xml(
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


def create_yolo_dataset(root: Path, sample_count: int) -> None:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(sample_count):
        image_name = f"sample_{idx:02d}.png"
        label_name = f"sample_{idx:02d}.txt"
        create_image(images_dir / image_name, color=(idx, idx, idx), size=(100, 100))
        (labels_dir / label_name).write_text(
            "0 0.500000 0.500000 0.400000 0.400000\n",
            encoding="utf-8",
        )

    (root / "classes.txt").write_text("object\n", encoding="utf-8")


def create_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
