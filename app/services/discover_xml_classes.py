"""扫描目录下全部 Pascal VOC XML，收集唯一类名并统计出现频次。"""

from pathlib import Path
from xml.etree import ElementTree

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import DiscoverXmlClassesRequest, DiscoverXmlClassesResponse


def run_discover_xml_classes(request: DiscoverXmlClassesRequest) -> DiscoverXmlClassesResponse:
    input_dir = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )

    xmls_dir = input_dir / request.xmls_dir_name
    if not xmls_dir.exists() or not xmls_dir.is_dir():
        raise ValueError(f"xmls_dir does not exist or is not a directory: {xmls_dir}")

    iterator = xmls_dir.rglob("*.xml") if request.recursive else xmls_dir.glob("*.xml")
    xml_paths = sorted([p for p in iterator if p.is_file()])

    class_counts: dict[str, int] = {}
    parse_errors = 0

    for xml_path in xml_paths:
        try:
            root = ElementTree.parse(xml_path).getroot()
        except ElementTree.ParseError:
            parse_errors += 1
            continue

        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if not name:
                continue
            difficult = (obj.findtext("difficult") or "").strip().lower()
            if not request.include_difficult and difficult in {"1", "true"}:
                continue
            class_counts[name] = class_counts.get(name, 0) + 1

    classes_sorted = sorted(class_counts.keys())

    return DiscoverXmlClassesResponse(
        input_dir=str(input_dir),
        xmls_dir=str(xmls_dir),
        total_xml_files=len(xml_paths),
        parse_errors=parse_errors,
        class_names=classes_sorted,
        class_counts=class_counts,
        total_classes=len(classes_sorted),
    )
