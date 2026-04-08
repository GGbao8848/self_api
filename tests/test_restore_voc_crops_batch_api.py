"""Tests for restore-voc-crops-batch."""

from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image, create_voc_xml


def test_restore_voc_crops_batch_one_crop(client: TestClient, case_dir: Path) -> None:
    orig_img = case_dir / "orig" / "images"
    orig_xml = case_dir / "orig" / "xmls"
    crop_img = case_dir / "crop" / "images"
    crop_xml = case_dir / "crop" / "xmls"
    out_dir = case_dir / "out"
    for d in (orig_img, orig_xml, crop_img, crop_xml):
        d.mkdir(parents=True)

    create_image(orig_img / "1_5.jpg", (100, 100, 100), (200, 80))
    create_voc_xml(
        orig_xml / "1_5.xml",
        "1_5.jpg",
        (200, 80),
        [("a", (10, 10, 30, 30))],
    )
    create_image(crop_img / "1_5_cx40_cy40_S40.jpg", (255, 0, 0), (20, 20))
    create_voc_xml(
        crop_xml / "1_5_cx40_cy40_S40.xml",
        "1_5_cx40_cy40_S40.jpg",
        (20, 20),
        [("a", (2, 2, 18, 18))],
    )

    r = client.post(
        "/api/v1/preprocess/restore-voc-crops-batch",
        json={
            "original_images_dir": str(orig_img),
            "original_xmls_dir": str(orig_xml),
            "edited_crops_images_dir": str(crop_img),
            "edited_crops_xmls_dir": str(crop_xml),
            "output_dir": str(out_dir),
            "recursive": False,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["originals_processed"] == 1
    assert (out_dir / "images" / "1_5.jpg").is_file()
    assert (out_dir / "xmls" / "1_5.xml").is_file()


def test_parse_voc_bar_stem() -> None:
    from app.services.voc_crop_restore import parse_voc_bar_crop_stem, region_xywh_from_cx_cy_s

    assert parse_voc_bar_crop_stem("1_5_cx5767_cy563_S1126") == ("1_5", 5767, 563, 1126)
    x, y, w, h = region_xywh_from_cx_cy_s(5767, 563, 1126)
    assert (x, y, w, h) == (5204, 0, 1126, 1126)
