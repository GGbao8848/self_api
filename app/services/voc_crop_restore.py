"""Shared helpers for VOC crop restore / paste-back (geometry + voc-bar filename parsing)."""

from __future__ import annotations

import re
from pathlib import Path

from app.services.yolo_sliding_window import _xyxy_to_yolo, _yolo_to_xyxy

# voc-bar-crop 产出：{stem}_cx{cx}_cy{cy}_S{S}
_CROP_STEM_RE = re.compile(r"^(.+)_cx(\d+)_cy(\d+)_S(\d+)$")


def parse_voc_bar_crop_stem(stem: str) -> tuple[str, int, int, int] | None:
    """从裁剪文件名 stem 解析原图 stem、窗口中心 cx/cy、边长 S。无法解析则返回 None。"""
    m = _CROP_STEM_RE.match(stem)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def region_xywh_from_cx_cy_s(cx: int, cy: int, s: int) -> tuple[int, int, int, int]:
    """与 voc_bar_crop 命名约定一致：cx = left + S//2, cy = top + S//2（整数）。"""
    x = cx - s // 2
    y = cy - s // 2
    return x, y, s, s


def region_overlaps_box(
    rx: int,
    ry: int,
    rw: int,
    rh: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> bool:
    ix1 = max(rx, x1)
    iy1 = max(ry, y1)
    ix2 = min(rx + rw, x2)
    iy2 = min(ry + rh, y2)
    return ix2 > ix1 and iy2 > iy1


def clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    nx1 = int(max(0, min(x1, w - 1)))
    ny1 = int(max(0, min(y1, h - 1)))
    nx2 = int(max(0, min(x2, w - 1)))
    ny2 = int(max(0, min(y2, h - 1)))
    if nx2 <= nx1:
        nx2 = min(w - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h - 1, ny1 + 1)
    return nx1, ny1, nx2, ny2


def map_small_voc_to_large(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    sw: int,
    sh: int,
    region_x: int,
    region_y: int,
    region_w: int,
    region_h: int,
    lw: int,
    lh: int,
) -> tuple[int, int, int, int]:
    sx1 = xmin * region_w / sw + region_x
    sy1 = ymin * region_h / sh + region_y
    sx2 = xmax * region_w / sw + region_x
    sy2 = ymax * region_h / sh + region_y
    return clip_xyxy(sx1, sy1, sx2, sy2, lw, lh)


def map_small_yolo_to_large(
    cls_id: int,
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    sw: int,
    sh: int,
    region_x: int,
    region_y: int,
    region_w: int,
    region_h: int,
    lw: int,
    lh: int,
) -> tuple[int, float, float, float, float]:
    x1, y1, x2, y2 = _yolo_to_xyxy(xc, yc, bw, bh, sw, sh)
    lx1, ly1, lx2, ly2 = map_small_voc_to_large(
        int(x1), int(y1), int(x2), int(y2), sw, sh, region_x, region_y, region_w, region_h, lw, lh
    )
    nxc, nyc, nbw, nbh = _xyxy_to_yolo(float(lx1), float(ly1), float(lx2), float(ly2), lw, lh)
    return cls_id, nxc, nyc, nbw, nbh


def resolve_original_image(original_images_dir: Path, stem: str) -> Path | None:
    from app.services.xml_to_yolo import _resolve_image_path

    return _resolve_image_path(original_images_dir, None, stem)
