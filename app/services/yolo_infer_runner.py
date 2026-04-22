from __future__ import annotations

import argparse
import json
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMG_EXTS else []
    if not path.is_dir():
        return []
    iterator = path.rglob("*") if recursive else path.glob("*")
    return [p for p in iterator if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _save_yolo_txt(label_path: Path, boxes_xyxy, cls_ids, img_w: int, img_h: int) -> None:
    lines: list[str] = []
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, cls_ids):
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        xc = min(max(((x1 + x2) / 2.0) / img_w, 0.0), 1.0)
        yc = min(max(((y1 + y2) / 2.0) / img_h, 0.0), 1.0)
        w = min(max((x2 - x1) / img_w, 0.0), 1.0)
        h = min(max((y2 - y1) / img_h, 0.0), 1.0)
        lines.append(f"{int(c)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _emit_progress(*, current: int, total: int, stage: str, message: str) -> None:
    sys.stdout.write(
        "__SELF_API_PROGRESS__"
        + json.dumps(
            {
                "current": current,
                "total": total,
                "stage": stage,
                "message": message,
                "unit": "image",
                "indeterminate": False,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--classes", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--recursive", type=int, default=1)
    parser.add_argument("--save-labels", type=int, default=1)
    parser.add_argument("--save-no-detect", type=int, default=1)
    parser.add_argument("--add-conf-prefix", type=int, default=1)
    parser.add_argument("--draw-label", type=int, default=1)
    parser.add_argument("--overwrite", type=int, default=1)
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    project = Path(args.project).expanduser().resolve()
    output_dir = project / args.name
    sources = [Path(s).expanduser().resolve() for s in args.source]
    if output_dir.exists():
        if not bool(args.overwrite):
            raise RuntimeError(f"infer output dir already exists: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "result").mkdir(parents=True, exist_ok=True)
    if bool(args.save_labels):
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    if bool(args.save_no_detect):
        (output_dir / "no_detect").mkdir(parents=True, exist_ok=True)

    run_args_path = output_dir / "args.yaml"
    summary_path = output_dir / "summary.json"
    run_args_path.write_text(
        yaml.safe_dump(
            {
                "model_path": str(model_path),
                "sources": [str(s) for s in sources],
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "classes": args.classes,
                "device": args.device or None,
                "recursive": bool(args.recursive),
                "save_labels": bool(args.save_labels),
                "save_no_detect": bool(args.save_no_detect),
                "add_conf_prefix": bool(args.add_conf_prefix),
                "draw_label": bool(args.draw_label),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    images: list[Path] = []
    seen: set[str] = set()
    for src in sources:
        for p in _iter_images(src, bool(args.recursive)):
            k = str(p)
            if k not in seen:
                seen.add(k)
                images.append(p)
    if not images:
        raise RuntimeError("no images found from input sources")

    cls_filter = None
    if args.classes.strip():
        cls_filter = [int(x) for x in args.classes.split(",") if x.strip()]

    model = YOLO(str(model_path), task="detect")
    detected_images = 0
    no_detect_images = 0
    labels_written = 0
    for img_path in images:
        _emit_progress(
            current=detected_images + no_detect_images,
            total=len(images),
            stage="infer_images",
            message=f"running inference on {img_path.name}",
        )
        results = model.predict(
            source=str(img_path),
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            classes=cls_filter,
            save=False,
            verbose=False,
            device=(args.device or None),
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            no_detect_images += 1
            if bool(args.save_no_detect):
                shutil.copy2(img_path, output_dir / "no_detect" / img_path.name)
            _emit_progress(
                current=detected_images + no_detect_images,
                total=len(images),
                stage="infer_images",
                message=f"processed {detected_images + no_detect_images}/{len(images)} images",
            )
            continue

        detected_images += 1
        conf_prefix = ""
        if bool(args.add_conf_prefix) and r.boxes.conf is not None and len(r.boxes.conf) > 0:
            conf_prefix = f"Conf-{float(r.boxes.conf.max().cpu().item()):.6f}_"
        out_name = f"{conf_prefix}{img_path.name}"
        shutil.copy2(img_path, output_dir / "images" / out_name)
        plotted = r.plot(conf=bool(args.draw_label), labels=bool(args.draw_label))
        cv2.imwrite(str(output_dir / "result" / out_name), plotted)

        if bool(args.save_labels):
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            cls_ids = np.asarray(r.boxes.cls.cpu().numpy()).astype(int)
            _save_yolo_txt(
                output_dir / "labels" / f"{conf_prefix}{img_path.stem}.txt",
                boxes_xyxy,
                cls_ids,
                int(plotted.shape[1]),
                int(plotted.shape[0]),
            )
            labels_written += 1
        _emit_progress(
            current=detected_images + no_detect_images,
            total=len(images),
            stage="infer_images",
            message=f"processed {detected_images + no_detect_images}/{len(images)} images",
        )

    summary = {
        "status": "ok",
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "total_images": len(images),
        "detected_images": detected_images,
        "no_detect_images": no_detect_images,
        "result_images": detected_images,
        "labels_written": labels_written,
        "classes_filter": cls_filter,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
