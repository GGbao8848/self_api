from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_image, create_voc_xml


def test_annotate_visualize_yolo_empty_classes_file_shows_id(client: TestClient, case_dir: Path) -> None:
    root = case_dir / "yolo_vis_empty_cf"
    images_dir = root / "images"
    out_dir = root / "viz_out"
    images_dir.mkdir(parents=True)
    (root / "labels").mkdir(parents=True)

    create_image(images_dir / "a.jpg", color=(40, 40, 40), size=(100, 100))
    (root / "labels" / "a.txt").write_text("2 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/annotate-visualize",
        json={
            "input_dir": str(root),
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
    out_dir = root / "viz_out"
    images_dir.mkdir(parents=True)
    (root / "labels").mkdir(parents=True)

    create_image(images_dir / "a.jpg", color=(40, 40, 40), size=(100, 100))
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/annotate-visualize",
        json={
            "input_dir": str(root),
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
    out_dir = root / "viz_out"
    images_dir.mkdir(parents=True)
    (root / "xmls").mkdir(parents=True)

    create_image(images_dir / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    create_voc_xml(
        root / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )

    response = client.post(
        "/api/v1/preprocess/annotate-visualize",
        json={
            "input_dir": str(root),
            "output_dir": str(out_dir),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "xml"
    assert data["written_images"] == 1
    assert (out_dir / "img_1.jpg").exists()


def test_annotate_visualize_prefers_yolo_when_labels_and_xmls_both_exist(
    client: TestClient,
    case_dir: Path,
) -> None:
    root = case_dir / "both_vis"
    images_dir = root / "images"
    labels_dir = root / "labels"
    xmls_dir = root / "xmls"
    out_dir = root / "viz_out"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    xmls_dir.mkdir(parents=True)

    create_image(images_dir / "a.jpg", color=(40, 40, 40), size=(100, 100))
    (labels_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    create_voc_xml(
        xmls_dir / "a.xml",
        filename="a.jpg",
        size=(100, 100),
        objects=[("xml_obj", (10, 20, 60, 80))],
    )

    response = client.post(
        "/api/v1/preprocess/annotate-visualize",
        json={
            "input_dir": str(root),
            "output_dir": str(out_dir),
            "classes": ["yolo_obj"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "yolo"
    assert data["annotation_dir"] == str(labels_dir)


def test_annotate_visualize_requires_labels_or_xmls(client: TestClient, case_dir: Path) -> None:
    root = case_dir / "missing_ann"
    (root / "images").mkdir(parents=True)
    response = client.post(
        "/api/v1/preprocess/annotate-visualize",
        json={
            "input_dir": str(root),
            "output_dir": str(root / "viz_out"),
        },
    )
    assert response.status_code == 400
