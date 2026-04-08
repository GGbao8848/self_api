"""Tests for voc-bar-crop preprocess API."""

from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image, create_voc_xml


def test_voc_bar_crop_horizontal_bar(client: TestClient, case_dir: Path) -> None:
    images_dir = case_dir / "images"
    xmls_dir = case_dir / "xmls"
    out_dir = case_dir / "out"
    images_dir.mkdir()
    xmls_dir.mkdir()
    create_image(images_dir / "wide.jpg", (200, 200, 200), (200, 100))
    create_voc_xml(
        xmls_dir / "wide.xml",
        "wide.jpg",
        (200, 100),
        [("bar", (40, 30, 160, 45))],
    )
    r = client.post(
        "/api/v1/preprocess/voc-bar-crop",
        json={
            "images_dir": str(images_dir),
            "xmls_dir": str(xmls_dir),
            "output_dir": str(out_dir),
            "recursive": False,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["generated_crops"] == 1
    assert data["processed_xml_files"] == 1
    # 横向长条：正方形边长 = 整图高度 100，而非标注框高 15
    with_crop = [d for d in data["details"] if d.get("crop_image")]
    assert len(with_crop) == 1
    assert with_crop[0].get("window_size") == 100
    crops = list((out_dir / "images").glob("*.jpg"))
    assert len(crops) == 1
    assert "_cx" in crops[0].name and crops[0].name.endswith("_S100.jpg")
    xml_out = (out_dir / "xmls" / crops[0].with_suffix(".xml").name)
    assert xml_out.is_file()


def test_voc_bar_crop_skips_vertical_box(client: TestClient, case_dir: Path) -> None:
    images_dir = case_dir / "images"
    xmls_dir = case_dir / "xmls"
    out_dir = case_dir / "out"
    images_dir.mkdir()
    xmls_dir.mkdir()
    create_image(images_dir / "v.jpg", (100, 100, 100), (100, 100))
    create_voc_xml(
        xmls_dir / "v.xml",
        "v.jpg",
        (100, 100),
        [("tall", (45, 20, 55, 80))],
    )
    r = client.post(
        "/api/v1/preprocess/voc-bar-crop",
        json={
            "images_dir": str(images_dir),
            "xmls_dir": str(xmls_dir),
            "output_dir": str(out_dir),
            "recursive": False,
        },
    )
    assert r.status_code == 200
    assert r.json()["generated_crops"] == 0
