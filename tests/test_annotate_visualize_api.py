from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image, create_voc_xml


def test_annotate_visualize_yolo_empty_classes_file_shows_id(client: TestClient, case_dir: Path) -> None:
    root = case_dir / "yolo_vis_empty_cf"
    images_dir = root / "images"
    labels_dir = root / "labels"
    out_dir = root / "viz_out"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    create_image(images_dir / "a.jpg", color=(40, 40, 40), size=(100, 100))
    (labels_dir / "a.txt").write_text("2 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/annotate-visualize",
        json={
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "xmls_dir": "",
            "output_dir": str(out_dir),
            "classes_file": "",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "yolo"
    assert data["written_images"] == 1


def test_annotate_visualize_yolo(client: TestClient, case_dir: Path) -> None:
    root = case_dir / "yolo_vis"
    images_dir = root / "images"
    labels_dir = root / "labels"
    out_dir = root / "viz_out"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    create_image(images_dir / "a.jpg", color=(40, 40, 40), size=(100, 100))
    (labels_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/annotate-visualize",
        json={
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "xmls_dir": "",
            "output_dir": str(out_dir),
            "classes": ["obj"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "yolo"
    assert data["written_images"] == 1
    assert data["skipped_images"] == 0
    assert (out_dir / "a.jpg").exists()


def test_annotate_visualize_xml(client: TestClient, case_dir: Path) -> None:
    root = case_dir / "xml_vis"
    images_dir = root / "images"
    xmls_dir = root / "xmls"
    out_dir = root / "viz_out"
    images_dir.mkdir(parents=True)
    xmls_dir.mkdir(parents=True)

    create_image(images_dir / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    create_voc_xml(
        xmls_dir / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )

    response = client.post(
        "/api/v1/preprocess/annotate-visualize",
        json={
            "images_dir": str(images_dir),
            "labels_dir": "",
            "xmls_dir": str(xmls_dir),
            "output_dir": str(out_dir),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "xml"
    assert data["written_images"] == 1
    assert (out_dir / "img_1.jpg").exists()


def test_annotate_visualize_mutual_exclusive(client: TestClient) -> None:
    response = client.post(
        "/api/v1/preprocess/annotate-visualize",
        json={
            "images_dir": "/tmp/x",
            "labels_dir": "/tmp/a",
            "xmls_dir": "/tmp/b",
            "output_dir": "/tmp/o",
        },
    )
    assert response.status_code == 422

    response = client.post(
        "/api/v1/preprocess/annotate-visualize",
        json={
            "images_dir": "/tmp/x",
            "labels_dir": "",
            "xmls_dir": "",
            "output_dir": "/tmp/o",
        },
    )
    assert response.status_code == 422
