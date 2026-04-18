import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError

from fastapi.testclient import TestClient

from app.services import task_manager
from tests.data_helpers import create_image, create_text_file, create_voc_xml, create_yolo_dataset


def _wait_task_done(
    client: TestClient,
    task_id: str,
    *,
    timeout_seconds: float = 8.0,
) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/api/v1/preprocess/tasks/{task_id}")
        assert response.status_code == 200
        task_data = response.json()
        if task_data["state"] in {"succeeded", "failed"}:
            return task_data
        time.sleep(0.05)
    raise AssertionError(f"task did not complete in {timeout_seconds} seconds: {task_id}")


def _wait_callback_done(
    client: TestClient,
    task_id: str,
    *,
    timeout_seconds: float = 8.0,
) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/api/v1/preprocess/tasks/{task_id}")
        assert response.status_code == 200
        task_data = response.json()
        if task_data["callback_state"] in {"succeeded", "failed"}:
            return task_data
        time.sleep(0.05)
    raise AssertionError(
        f"callback did not finish in {timeout_seconds} seconds: {task_id}"
    )


def test_xml_to_yolo_async_endpoint(client: TestClient, case_dir: Path) -> None:
    dataset_dir = case_dir / "voc_dataset"
    images_dir = dataset_dir / "images"
    xmls_dir = dataset_dir / "xmls"

    create_image(images_dir / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    create_voc_xml(
        xmls_dir / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )

    submit_resp = client.post(
        "/api/v1/preprocess/xml-to-yolo/async",
        json={"input_dir": str(dataset_dir)},
    )
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()
    assert submit_data["task_type"] == "xml_to_yolo"
    assert submit_data["callback_url"] is None

    task_data = _wait_task_done(client, submit_data["task_id"])
    assert task_data["state"] == "succeeded"
    assert task_data["result"]["converted_files"] == 1


def test_split_yolo_dataset_async_endpoint(client: TestClient, case_dir: Path) -> None:
    dataset_dir = case_dir / "yolo_dataset"
    output_dir = case_dir / "split_out"
    create_yolo_dataset(dataset_dir, sample_count=4)

    submit_resp = client.post(
        "/api/v1/preprocess/split-yolo-dataset/async",
        json={
            "input_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "mode": "train_only",
            "shuffle": False,
        },
    )
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()
    assert submit_data["task_type"] == "split_yolo_dataset"

    task_data = _wait_task_done(client, submit_data["task_id"])
    assert task_data["state"] == "succeeded"
    assert task_data["result"]["train_images"] == 4


def test_zip_folder_async_endpoint(client: TestClient, case_dir: Path) -> None:
    source_dir = case_dir / "source_pack"
    create_text_file(source_dir / "a.txt", "hello")
    create_text_file(source_dir / "nested" / "b.txt", "world")

    submit_resp = client.post(
        "/api/v1/preprocess/zip-folder/async",
        json={"input_dir": str(source_dir)},
    )
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()
    assert submit_data["task_type"] == "zip_folder"

    task_data = _wait_task_done(client, submit_data["task_id"])
    assert task_data["state"] == "succeeded"
    output_zip_path = Path(task_data["result"]["output_zip_path"])
    assert output_zip_path.exists()


def test_unzip_archive_async_endpoint(client: TestClient, case_dir: Path) -> None:
    archive_path = case_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("sample/x.txt", "x")

    output_dir = case_dir / "unzipped"
    submit_resp = client.post(
        "/api/v1/preprocess/unzip-archive/async",
        json={"archive_path": str(archive_path), "output_dir": str(output_dir)},
    )
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()
    assert submit_data["task_type"] == "unzip_archive"

    task_data = _wait_task_done(client, submit_data["task_id"])
    assert task_data["state"] == "succeeded"
    assert (output_dir / "sample" / "x.txt").read_text(encoding="utf-8") == "x"


def test_copy_path_async_endpoint(client: TestClient, case_dir: Path) -> None:
    source_file = case_dir / "one.txt"
    create_text_file(source_file, "payload")
    target_dir = case_dir / "target"

    submit_resp = client.post(
        "/api/v1/preprocess/copy-path/async",
        json={"source_path": str(source_file), "target_dir": str(target_dir)},
    )
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()
    assert submit_data["task_type"] == "copy_path"

    task_data = _wait_task_done(client, submit_data["task_id"])
    assert task_data["state"] == "succeeded"
    assert (target_dir / "one.txt").read_text(encoding="utf-8") == "payload"
    assert source_file.exists()
    assert source_file.read_text(encoding="utf-8") == "payload"


def test_move_path_async_endpoint(client: TestClient, case_dir: Path) -> None:
    source_file = case_dir / "one.txt"
    create_text_file(source_file, "payload")
    target_dir = case_dir / "target"

    submit_resp = client.post(
        "/api/v1/preprocess/move-path/async",
        json={"source_path": str(source_file), "target_dir": str(target_dir)},
    )
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()
    assert submit_data["task_type"] == "move_path"

    task_data = _wait_task_done(client, submit_data["task_id"])
    assert task_data["state"] == "succeeded"
    assert (target_dir / "one.txt").read_text(encoding="utf-8") == "payload"


def test_move_path_async_endpoint_callback_success(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    source_file = case_dir / "one.txt"
    create_text_file(source_file, "payload")
    target_dir = case_dir / "target"

    callback_calls: list[dict] = []

    def _fake_post_callback(callback_url: str, payload: dict, timeout: float) -> int:
        callback_calls.append(
            {"callback_url": callback_url, "payload": payload, "timeout": timeout}
        )
        return 204

    monkeypatch.setattr(task_manager, "_post_callback", _fake_post_callback)

    submit_resp = client.post(
        "/api/v1/preprocess/move-path/async",
        json={
            "source_path": str(source_file),
            "target_dir": str(target_dir),
            "callback_url": "http://127.0.0.1:9999/callback",
        },
    )
    assert submit_resp.status_code == 202
    task_id = submit_resp.json()["task_id"]

    task_data = _wait_task_done(client, task_id)
    assert task_data["state"] == "succeeded"
    task_data = _wait_callback_done(client, task_id)
    assert task_data["callback_state"] == "succeeded"
    assert task_data["callback_status_code"] == 204
    assert task_data["callback_error"] is None
    assert len(task_data["callback_events"]) == 1
    assert (target_dir / "one.txt").read_text(encoding="utf-8") == "payload"

    assert len(callback_calls) == 1
    assert callback_calls[0]["callback_url"] == "http://127.0.0.1:9999/callback"
    assert callback_calls[0]["payload"]["task_id"] == task_id
    assert callback_calls[0]["payload"]["task_type"] == "move_path"
    assert callback_calls[0]["payload"]["state"] == "succeeded"


def test_copy_path_async_endpoint_callback_success(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    source_file = case_dir / "one.txt"
    create_text_file(source_file, "payload")
    target_dir = case_dir / "target"

    callback_calls: list[dict] = []

    def _fake_post_callback(callback_url: str, payload: dict, timeout: float) -> int:
        callback_calls.append(
            {"callback_url": callback_url, "payload": payload, "timeout": timeout}
        )
        return 204

    monkeypatch.setattr(task_manager, "_post_callback", _fake_post_callback)

    submit_resp = client.post(
        "/api/v1/preprocess/copy-path/async",
        json={
            "source_path": str(source_file),
            "target_dir": str(target_dir),
            "callback_url": "http://127.0.0.1:9999/callback",
        },
    )
    assert submit_resp.status_code == 202
    task_id = submit_resp.json()["task_id"]

    task_data = _wait_task_done(client, task_id)
    assert task_data["state"] == "succeeded"
    task_data = _wait_callback_done(client, task_id)
    assert task_data["callback_state"] == "succeeded"
    assert task_data["callback_status_code"] == 204
    assert task_data["callback_error"] is None
    assert len(task_data["callback_events"]) == 1
    assert source_file.exists()
    assert (target_dir / "one.txt").read_text(encoding="utf-8") == "payload"

    assert len(callback_calls) == 1
    assert callback_calls[0]["callback_url"] == "http://127.0.0.1:9999/callback"
    assert callback_calls[0]["payload"]["task_id"] == task_id
    assert callback_calls[0]["payload"]["task_type"] == "copy_path"
    assert callback_calls[0]["payload"]["state"] == "succeeded"


def test_move_path_async_endpoint_callback_conflict_treated_as_success(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    source_file = case_dir / "one.txt"
    create_text_file(source_file, "payload")
    target_dir = case_dir / "target"

    callback_calls: list[dict] = []

    def _fake_post_callback(callback_url: str, payload: dict, timeout: float) -> int:
        callback_calls.append(
            {"callback_url": callback_url, "payload": payload, "timeout": timeout}
        )
        raise HTTPError(callback_url, 409, "Conflict", None, None)

    monkeypatch.setattr(task_manager, "_post_callback", _fake_post_callback)

    submit_resp = client.post(
        "/api/v1/preprocess/move-path/async",
        json={
            "source_path": str(source_file),
            "target_dir": str(target_dir),
            "callback_url": "http://127.0.0.1:9999/callback",
        },
    )
    assert submit_resp.status_code == 202
    task_id = submit_resp.json()["task_id"]

    task_data = _wait_task_done(client, task_id)
    assert task_data["state"] == "succeeded"
    task_data = _wait_callback_done(client, task_id)
    assert task_data["callback_state"] == "succeeded"
    assert task_data["callback_status_code"] == 409
    assert task_data["callback_error"] is None
    assert len(task_data["callback_events"]) == 1
    assert task_data["callback_events"][0]["success"] is True
    assert task_data["callback_events"][0]["status_code"] == 409
    assert task_data["callback_events"][0]["error"] is None

    assert len(callback_calls) == 1
    assert callback_calls[0]["callback_url"] == "http://127.0.0.1:9999/callback"
    assert callback_calls[0]["payload"]["task_id"] == task_id
    assert callback_calls[0]["payload"]["task_type"] == "move_path"
    assert callback_calls[0]["payload"]["state"] == "succeeded"


def test_yolo_sliding_window_crop_async_endpoint(
    client: TestClient,
    case_dir: Path,
) -> None:
    images_dir = case_dir / "images"
    labels_dir = case_dir / "labels"
    output_dir = case_dir / "data_crops"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    create_image(images_dir / "wide.png", color=(200, 10, 10), size=(16, 8))
    (labels_dir / "wide.txt").write_text(
        "0 0.250000 0.250000 0.250000 0.250000\n",
        encoding="utf-8",
    )

    submit_resp = client.post(
        "/api/v1/preprocess/yolo-sliding-window-crop/async",
        json={
            "input_dir": str(case_dir),
            "output_dir": str(output_dir),
            "min_vis_ratio": 0.5,
            "stride_ratio": 0.25,
            "ignore_vis_ratio": 0.05,
            "only_wide": True,
        },
    )
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()
    assert submit_data["task_type"] == "yolo_sliding_window_crop"

    task_data = _wait_task_done(client, submit_data["task_id"])
    assert task_data["state"] == "succeeded"
    assert task_data["result"]["generated_crops"] >= 1
