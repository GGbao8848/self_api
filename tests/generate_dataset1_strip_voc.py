#!/usr/bin/env python3
"""一次性脚本：生成横向长条图 VOC 数据集 dataset1（images + xmls）。"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

W, H = 20000, 819
NUM_IMAGES = 100
CLASSES = ("dog", "cat", "pig")
CLASS_COLOR = {
    "dog": (180, 120, 60),
    "cat": (140, 140, 160),
    "pig": (230, 180, 190),
}


def build_base_image(rng: random.Random) -> Image.Image:
    """由小图双线性放大，避免逐像素填充 20000×819。"""
    sw, sh = 250, 102
    thumb = Image.new("RGB", (sw, sh))
    px = thumb.load()
    seed = rng.randint(0, 10_000_000)
    for x in range(sw):
        for y in range(sh):
            px[x, y] = (
                (x * 17 + y * 3 + seed) % 256,
                (x * 11 + y * 19 + seed // 2) % 256,
                (x * 7 + y * 23 + seed // 3) % 256,
            )
    return thumb.resize((W, H), Image.Resampling.BILINEAR)


def sample_objects(rng: random.Random) -> list[tuple[str, tuple[int, int, int, int]]]:
    """每张图至少各 2 个 dog/cat/pig，再随机补充，横向分栏避免严重重叠。"""
    n_extra = rng.randint(6, 12)
    names = list(CLASSES) * 2 + [rng.choice(CLASSES) for _ in range(n_extra)]
    rng.shuffle(names)
    n = len(names)
    col_w = W // n
    objects: list[tuple[str, tuple[int, int, int, int]]] = []
    for i, cls in enumerate(names):
        margin_x = 8
        margin_y = 8
        x_lo = i * col_w + margin_x
        x_hi = (i + 1) * col_w - margin_x
        bw_max = min(420, x_hi - x_lo - 4)
        bh_max = min(380, H - 2 * margin_y)
        if bw_max < 80 or bh_max < 60:
            continue
        bw = rng.randint(80, bw_max)
        bh = rng.randint(60, bh_max)
        x0 = rng.randint(x_lo, max(x_lo, x_hi - bw))
        y0 = rng.randint(margin_y, H - margin_y - bh)
        xmin, ymin, xmax, ymax = x0, y0, x0 + bw, y0 + bh
        objects.append((cls, (xmin, ymin, xmax, ymax)))
    return objects


def write_voc_xml(path: Path, filename: str, objects: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(W)
    ET.SubElement(size, "height").text = str(H)
    ET.SubElement(size, "depth").text = "3"
    for cls, (xmin, ymin, xmax, ymax) in objects:
        obj_el = ET.SubElement(root, "object")
        ET.SubElement(obj_el, "name").text = cls
        ET.SubElement(obj_el, "difficult").text = "0"
        bb = ET.SubElement(obj_el, "bndbox")
        ET.SubElement(bb, "xmin").text = str(xmin)
        ET.SubElement(bb, "ymin").text = str(ymin)
        ET.SubElement(bb, "xmax").text = str(xmax)
        ET.SubElement(bb, "ymax").text = str(ymax)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    root = Path(__file__).resolve().parent / "dataset1"
    images_dir = root / "images"
    xmls_dir = root / "xmls"
    images_dir.mkdir(parents=True, exist_ok=True)
    xmls_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(NUM_IMAGES):
        rng = random.Random(42_000 + idx)
        stem = f"strip_{idx:03d}"
        jpg_name = f"{stem}.jpg"
        img = build_base_image(rng)
        draw = ImageDraw.Draw(img)
        objects = sample_objects(rng)
        for cls, (xmin, ymin, xmax, ymax) in objects:
            c = CLASS_COLOR[cls]
            draw.rectangle([xmin, ymin, xmax, ymax], outline=c, width=4)
            draw.rectangle([xmin + 3, ymin + 3, xmax - 3, ymax - 3], fill=c, outline=c)
        img.save(images_dir / jpg_name, format="JPEG", quality=88, optimize=True)
        write_voc_xml(xmls_dir / f"{stem}.xml", jpg_name, objects)
        if (idx + 1) % 10 == 0:
            print(f"written {idx + 1}/{NUM_IMAGES}")

    (root / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    print(f"done -> {root}")


if __name__ == "__main__":
    main()
