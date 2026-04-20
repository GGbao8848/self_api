from pathlib import Path

from fastapi.testclient import TestClient

from tests.data_helpers import create_voc_xml


def test_discover_xml_classes_basic(client: TestClient, case_dir: Path) -> None:
    dataset_dir = case_dir / "voc_dataset"
    xmls_dir = dataset_dir / "xmls"

    create_voc_xml(
        xmls_dir / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("louyou1", (10, 20, 60, 80)), ("louyou2", (30, 40, 90, 90))],
    )
    create_voc_xml(
        xmls_dir / "img_2.xml",
        filename="img_2.jpg",
        size=(100, 100),
        objects=[("louyou1", (1, 1, 30, 30)), ("other", (40, 40, 80, 80))],
    )

    response = client.post(
        "/api/v1/preprocess/discover-xml-classes",
        json={"input_dir": str(dataset_dir)},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total_xml_files"] == 2
    assert data["parse_errors"] == 0
    assert data["total_classes"] == 3
    assert sorted(data["class_names"]) == ["louyou1", "louyou2", "other"]
    assert data["class_counts"]["louyou1"] == 2
    assert data["class_counts"]["louyou2"] == 1
    assert data["class_counts"]["other"] == 1


def test_discover_xml_classes_skip_difficult_by_default(
    client: TestClient, case_dir: Path
) -> None:
    dataset_dir = case_dir / "voc_dataset"
    xmls_dir = dataset_dir / "xmls"
    xml_path = xmls_dir / "img_1.xml"

    create_voc_xml(
        xml_path,
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )

    text = xml_path.read_text(encoding="utf-8")
    text = text.replace(
        "<difficult>0</difficult>",
        "<difficult>1</difficult>",
        1,
    )
    xml_path.write_text(text, encoding="utf-8")

    response = client.post(
        "/api/v1/preprocess/discover-xml-classes",
        json={"input_dir": str(dataset_dir)},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_xml_files"] == 1
    assert data["total_classes"] == 0

    response_include = client.post(
        "/api/v1/preprocess/discover-xml-classes",
        json={"input_dir": str(dataset_dir), "include_difficult": True},
    )
    assert response_include.status_code == 200
    data_include = response_include.json()
    assert data_include["total_classes"] == 1
    assert data_include["class_counts"]["cat"] == 1


def test_discover_xml_classes_missing_dir_returns_400(
    client: TestClient, case_dir: Path
) -> None:
    dataset_dir = case_dir / "no_xmls"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/api/v1/preprocess/discover-xml-classes",
        json={"input_dir": str(dataset_dir)},
    )
    assert response.status_code == 400
    assert "xmls_dir" in response.json()["detail"]
