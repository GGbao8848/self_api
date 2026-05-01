import zipfile

from app.agent.sessions import agent_session_store
from app.agent.types import LLMToolDecision
from app.core.config import get_settings
from tests.data_helpers import create_image, create_text_file, create_voc_xml, create_yolo_dataset


def test_agent_chat_requires_provider(client, monkeypatch, isolated_runtime) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "")
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_MODEL", "")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "")
    monkeypatch.setenv("SELF_API_OPENAI_API_KEY", "")
    monkeypatch.setenv("SELF_API_OPENROUTER_API_KEY", "")
    get_settings.cache_clear()

    response = client.post("/api/v1/agent/chat", json={"message": "你好"})

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "requires_provider"
    assert data["session_id"]
    assert data["run_id"]
    assert data["tool_calls"] == []

    run_response = client.get(f"/api/v1/agent/runs/{data['run_id']}")
    assert run_response.status_code == 200
    assert run_response.json()["run_id"] == data["run_id"]

    session_response = client.get(f"/api/v1/agent/sessions/{data['session_id']}")
    assert session_response.status_code == 200
    assert session_response.json()["runs"][0]["run_id"] == data["run_id"]
    assert session_response.json()["runs"][0]["user_message"] == "你好"


def test_agent_sessions_list(client, monkeypatch, isolated_runtime) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "gemma4:e4b")
    from app.agent import runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "request_tool_decision",
        lambda **_: LLMToolDecision(action="respond", message="可以执行工具"),
    )
    get_settings.cache_clear()

    response = client.post("/api/v1/agent/chat", json={"message": "列出能力"})
    assert response.status_code == 200

    sessions_response = client.get("/api/v1/agent/sessions")
    assert sessions_response.status_code == 200
    data = sessions_response.json()
    assert len(data) == 1
    assert data[0]["preview"] == "列出能力"
    assert data[0]["messageCount"] == 1


def test_agent_empty_session_returns_empty_runs(client, isolated_runtime) -> None:
    agent_session_store.clear()

    response = client.get("/api/v1/agent/sessions/nonexistent-session")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "nonexistent-session"
    assert data["runs"] == []


def test_agent_chat_accepts_configured_ollama(client, monkeypatch, isolated_runtime) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "gemma4:e4b")
    from app.agent import runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "request_tool_decision",
        lambda **_: LLMToolDecision(action="respond", message="可执行 4 个工具"),
    )
    get_settings.cache_clear()

    response = client.post("/api/v1/agent/chat", json={"message": "列出能力"})

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["provider"] == "ollama"
    assert data["model"] == "gemma4:e4b"


def test_agent_tools_list(client, isolated_runtime) -> None:
    response = client.get("/api/v1/agent/tools")

    assert response.status_code == 200
    data = response.json()
    names = {item["name"] for item in data["items"]}
    assert data["total"] == len(data["items"])
    assert "scan-yolo-label-indices" in names
    assert "rewrite-yolo-label-indices" in names
    assert "xml-to-yolo" in names
    assert "yolo-sliding-window-crop" in names
    assert "yolo-augment" in names
    assert "annotate-visualize" in names
    assert "discover-leaf-dirs" in names
    assert "clean-nested-dataset-flat" in names
    assert "aggregate-nested-dataset" in names
    assert "build-yolo-yaml" in names
    assert "zip-folder" in names
    assert "unzip-archive" in names
    assert "move-path" in names
    assert "copy-path" in names
    assert "reset-yolo-label-index" in names
    assert "voc-bar-crop" in names
    assert "restore-voc-crops-batch" in names
    assert "publish-incremental-yolo-dataset" in names
    assert "publish-yolo-dataset" in names
    hints = {item["name"]: item["argument_hint"] for item in data["items"]}
    descriptions = {item["name"]: item["description"] for item in data["items"]}
    assert hints["build-yolo-yaml"]
    assert "output_yaml_path" in hints["build-yolo-yaml"]
    assert hints["publish-incremental-yolo-dataset"] == "{last_yaml, local_paths}"
    assert "remote_target" in hints["publish-yolo-dataset"]
    assert "archive_path" in hints["unzip-archive"]
    assert "edited_crops_images_dir" in hints["restore-voc-crops-batch"]
    assert descriptions["xml-to-yolo"] == "将 Pascal VOC XML 标注转换为 YOLO 标签。"
    assert descriptions["build-yolo-yaml"] == "根据数据集根目录生成 YOLO 的 data.yaml。"
    assert "yolo-train" not in names
    assert "remote-sbatch-yolo-train" not in names


def test_agent_executes_structured_scan_tool(client, case_dir, isolated_runtime) -> None:
    labels_dir = case_dir / "dataset" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text(
        "0 0.1 0.2 0.3 0.4\n2 0.2 0.3 0.4 0.5\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "scan labels",
            "tool_name": "scan-yolo-label-indices",
            "tool_arguments": {"input_dir": str(case_dir / "dataset")},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "scan-yolo-label-indices"
    assert data["tool_calls"][0]["error"] is None
    assert data["tool_calls"][0]["result"]["indices"] == [
        {"index": 0, "count": 1},
        {"index": 2, "count": 1},
    ]


def test_agent_routes_chinese_scan_request(client, case_dir, monkeypatch, isolated_runtime) -> None:
    from app.agent import runtime as runtime_module

    labels_dir = case_dir / "dataset" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text("3 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "request_tool_decision", lambda **_: LLMToolDecision(
        action="execute",
        tool_name="scan-yolo-label-indices",
        tool_arguments={"input_dir": str(case_dir / "dataset")},
    ))

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"查看标签索引 {case_dir / 'dataset'}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "scan-yolo-label-indices"
    assert data["tool_calls"][0]["result"]["indices"] == [{"index": 3, "count": 1}]


def test_agent_reuses_prior_tool_path_for_follow_up_request(
    client,
    case_dir,
    monkeypatch,
    isolated_runtime,
) -> None:
    from app.agent import runtime as runtime_module

    dataset_dir = case_dir / "voc_followup_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )

    def fake_request_tool_decision(**kwargs):
        message = kwargs["message"]
        if "转yolo" in message:
            return LLMToolDecision(
                action="execute",
                tool_name="xml-to-yolo",
                tool_arguments={"input_dir": str(dataset_dir)},
            )
        assert "Recent session resources:" in (kwargs.get("conversation_context") or "")
        return LLMToolDecision(
            action="execute",
            tool_name="scan-yolo-label-indices",
            tool_arguments={},
        )

    monkeypatch.setattr(runtime_module, "request_tool_decision", fake_request_tool_decision)

    first_response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"{dataset_dir} 转yolo"},
    )

    assert first_response.status_code == 200
    first_data = first_response.json()
    assert first_data["final_state"] == "completed"

    second_response = client.post(
        "/api/v1/agent/chat",
        json={
            "session_id": first_data["session_id"],
            "message": "看看索引",
        },
    )

    assert second_response.status_code == 200
    second_data = second_response.json()
    assert second_data["final_state"] == "completed"
    assert second_data["tool_calls"][0]["name"] == "scan-yolo-label-indices"
    assert second_data["tool_calls"][0]["arguments"]["input_dir"] == str(dataset_dir / "labels")
    assert second_data["tool_calls"][0]["result"]["indices"] == [{"index": 0, "count": 1}]


def test_agent_reuses_dataset_root_for_split_after_scan_follow_up(
    client,
    case_dir,
    monkeypatch,
    isolated_runtime,
) -> None:
    from app.agent import runtime as runtime_module

    dataset_dir = case_dir / "split_followup_dataset"
    create_yolo_dataset(dataset_dir, sample_count=4)

    def fake_request_tool_decision(**kwargs):
        message = kwargs["message"]
        if "看看索引" in message:
            return LLMToolDecision(
                action="execute",
                tool_name="scan-yolo-label-indices",
                tool_arguments={"input_dir": str(dataset_dir / "labels")},
            )
        assert "dataset_root" in (kwargs.get("conversation_context") or "")
        return LLMToolDecision(
            action="execute",
            tool_name="split-yolo-dataset",
            tool_arguments={"mode": "train_only"},
        )

    monkeypatch.setattr(runtime_module, "request_tool_decision", fake_request_tool_decision)

    first_response = client.post(
        "/api/v1/agent/chat",
        json={"message": "看看索引"},
    )

    assert first_response.status_code == 200
    first_data = first_response.json()
    assert first_data["final_state"] == "completed"
    assert first_data["tool_calls"][0]["arguments"]["input_dir"] == str(dataset_dir / "labels")

    second_response = client.post(
        "/api/v1/agent/chat",
        json={
            "session_id": first_data["session_id"],
            "message": "划分数据集 train only",
        },
    )

    assert second_response.status_code == 200
    second_data = second_response.json()
    assert second_data["final_state"] == "completed"
    assert second_data["tool_calls"][0]["name"] == "split-yolo-dataset"
    assert second_data["tool_calls"][0]["arguments"]["input_dir"] == str(dataset_dir)
    assert second_data["tool_calls"][0]["result"]["state"] == "succeeded"


def test_agent_routes_chinese_rewrite_request(client, case_dir, monkeypatch, isolated_runtime) -> None:
    from app.agent import runtime as runtime_module

    labels_dir = case_dir / "dataset" / "labels"
    labels_dir.mkdir(parents=True)
    label_file = labels_dir / "a.txt"
    label_file.write_text(
        "0 0.1 0.2 0.3 0.4\n2 0.2 0.3 0.4 0.5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_module, "request_tool_decision", lambda **_: LLMToolDecision(
        action="execute",
        tool_name="rewrite-yolo-label-indices",
        tool_arguments={"input_dir": str(case_dir / "dataset"), "default_target_index": 1},
    ))

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"把标签索引全部改成1 {case_dir / 'dataset'}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "rewrite-yolo-label-indices"
    assert data["tool_calls"][0]["result"]["changed_lines"] == 2
    assert label_file.read_text(encoding="utf-8").splitlines() == [
        "1 0.1 0.2 0.3 0.4",
        "1 0.2 0.3 0.4 0.5",
    ]


def test_agent_tool_missing_input_dir_returns_failed_state(client, isolated_runtime) -> None:
    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "scan labels",
            "tool_name": "scan-yolo-label-indices",
            "tool_arguments": {},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "failed"
    assert data["tool_calls"][0]["error"]


def test_agent_executes_structured_xml_to_yolo_tool(client, case_dir, isolated_runtime) -> None:
    dataset_dir = case_dir / "voc_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "xml to yolo",
            "tool_name": "xml-to-yolo",
            "tool_arguments": {"input_dir": str(dataset_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_result = data["tool_calls"][0]["result"]
    assert data["tool_calls"][0]["name"] == "xml-to-yolo"
    assert tool_result["state"] == "succeeded"
    assert tool_result["result"]["converted_files"] == 1
    assert tool_result["result"]["details_count"] == 1
    assert "details" not in tool_result["result"]
    assert (dataset_dir / "labels" / "img_1.txt").exists()


def test_agent_routes_chinese_xml_to_yolo_request(client, case_dir, monkeypatch, isolated_runtime) -> None:
    from app.agent import runtime as runtime_module

    dataset_dir = case_dir / "voc_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )
    monkeypatch.setattr(runtime_module, "request_tool_decision", lambda **_: LLMToolDecision(
        action="execute",
        tool_name="xml-to-yolo",
        tool_arguments={"input_dir": str(dataset_dir)},
    ))

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"xml转yolo {dataset_dir}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "xml-to-yolo"
    assert data["tool_calls"][0]["result"]["state"] == "succeeded"


def test_agent_routes_chinese_split_request(client, case_dir, monkeypatch, isolated_runtime) -> None:
    from app.agent import runtime as runtime_module

    dataset_dir = case_dir / "yolo_dataset"
    output_dir = case_dir / "yolo_dataset_split"
    create_yolo_dataset(dataset_dir, sample_count=5)
    monkeypatch.setattr(runtime_module, "request_tool_decision", lambda **_: LLMToolDecision(
        action="execute",
        tool_name="split-yolo-dataset",
        tool_arguments={"input_dir": str(dataset_dir), "output_dir": str(output_dir), "mode": "train_val", "train_ratio": 0.8, "val_ratio": 0.2, "test_ratio": 0.0},
    ))

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"划分数据集 8:2 {dataset_dir}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "split-yolo-dataset"
    tool_result = data["tool_calls"][0]["result"]
    assert tool_result["state"] == "succeeded"
    assert tool_result["result"]["mode"] == "train_val"
    assert tool_result["result"]["train_images"] == 4
    assert tool_result["result"]["val_images"] == 1
    assert tool_result["result"]["test_images"] == 0
    assert tool_result["result"]["details_count"] == 5
    assert (output_dir / "train" / "images").exists()


def test_agent_executes_structured_yolo_sliding_window_crop_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "crop_dataset"
    create_yolo_dataset(dataset_dir, sample_count=1)

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "crop dataset",
            "tool_name": "yolo-sliding-window-crop",
            "tool_arguments": {"input_dir": str(dataset_dir), "only_wide": False},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "yolo-sliding-window-crop"
    assert tool_call["arguments"]["output_dir"] == f"{dataset_dir}_yolo-sliding-window-crop"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["generated_crops"] == 1


def test_agent_executes_structured_yolo_augment_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "augment_dataset"
    create_yolo_dataset(dataset_dir, sample_count=1)

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "augment dataset",
            "tool_name": "yolo-augment",
            "tool_arguments": {
                "input_dir": str(dataset_dir),
                "vertical_flip": False,
                "brightness_up": False,
                "brightness_down": False,
                "contrast_up": False,
                "contrast_down": False,
                "gaussian_blur": False,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "yolo-augment"
    assert tool_call["arguments"]["output_dir"] == f"{dataset_dir}_aug"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["generated_images"] == 1


def test_agent_executes_structured_annotate_visualize_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "visualize_dataset"
    create_yolo_dataset(dataset_dir, sample_count=1)

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "visualize dataset",
            "tool_name": "annotate-visualize",
            "tool_arguments": {"input_dir": str(dataset_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "annotate-visualize"
    assert tool_call["arguments"]["output_dir"] == f"{dataset_dir}_visualized"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["written_images"] == 1


def test_agent_executes_structured_clean_nested_dataset_flat_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "nested_raw"
    leaf = dataset_dir / "session_a"
    create_image(leaf / "images" / "a.jpg", color=(255, 0, 0), size=(16, 16))
    create_voc_xml(
        leaf / "xmls" / "a.xml",
        filename="a.jpg",
        size=(16, 16),
        objects=[("cat", (2, 2, 8, 8))],
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "flatten dataset",
            "tool_name": "clean-nested-dataset-flat",
            "tool_arguments": {"input_dir": str(dataset_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "clean-nested-dataset-flat"
    assert tool_call["arguments"]["output_dir"] == f"{dataset_dir}_cleaned_flat"
    assert tool_call["arguments"]["flatten"] is True
    assert tool_call["arguments"]["include_backgrounds"] is False
    assert tool_call["arguments"]["pairing_mode"] == "images_xmls_subfolders"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["labeled_images"] == 1


def test_agent_executes_structured_discover_leaf_dirs_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    root_dir = case_dir / "leaf_root"
    create_image(root_dir / "group_a" / "leaf_1" / "a.jpg", color=(255, 0, 0), size=(16, 16))
    create_image(root_dir / "group_b" / "b.jpg", color=(0, 255, 0), size=(16, 16))

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "discover leaf dirs",
            "tool_name": "discover-leaf-dirs",
            "tool_arguments": {"input_dir": str(root_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "discover-leaf-dirs"
    assert tool_call["result"]["total_leaf_dirs"] == 2
    assert tool_call["result"]["leaf_dirs"] == [
        str(root_dir / "group_a" / "leaf_1"),
        str(root_dir / "group_b"),
    ]


def test_agent_executes_structured_aggregate_nested_dataset_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    root_dir = case_dir / "cleaned_fragments"
    fragment_dir = root_dir / "session_a"
    create_image(fragment_dir / "images" / "a.jpg", color=(255, 0, 0), size=(16, 16))
    (fragment_dir / "labels").mkdir(parents=True, exist_ok=True)
    (fragment_dir / "labels" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (fragment_dir / "classes.txt").write_text("cat\n", encoding="utf-8")

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "aggregate dataset",
            "tool_name": "aggregate-nested-dataset",
            "tool_arguments": {"input_dir": str(root_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "aggregate-nested-dataset"
    assert tool_call["arguments"]["output_dir"] == f"{root_dir}_dataset"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["aggregated_images"] == 1


def test_agent_executes_structured_zip_folder_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    source_dir = case_dir / "source_pack"
    create_text_file(source_dir / "a.txt", "hello")
    create_text_file(source_dir / "nested" / "b.txt", "world")

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "zip folder",
            "tool_name": "zip-folder",
            "tool_arguments": {"input_dir": str(source_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    tool_call = data["tool_calls"][0]
    assert data["final_state"] == "completed"
    assert tool_call["name"] == "zip-folder"
    assert tool_call["arguments"]["output_zip_path"] == f"{source_dir}.zip"
    assert tool_call["result"]["packed_files"] == 2
    with zipfile.ZipFile(tool_call["result"]["output_zip_path"], mode="r") as zipf:
        assert sorted(zipf.namelist()) == [
            f"{source_dir.name}/a.txt",
            f"{source_dir.name}/nested/b.txt",
        ]


def test_agent_executes_structured_unzip_archive_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    archive_path = case_dir / "sample.zip"
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("sample/x.txt", "x")

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "unzip archive",
            "tool_name": "unzip-archive",
            "tool_arguments": {"archive_path": str(archive_path)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    tool_call = data["tool_calls"][0]
    assert data["final_state"] == "completed"
    assert tool_call["name"] == "unzip-archive"
    assert tool_call["arguments"]["output_dir"] == str(case_dir / "sample")
    assert tool_call["result"]["extracted_files"] == 1
    assert (case_dir / "sample" / "sample" / "x.txt").read_text(encoding="utf-8") == "x"


def test_agent_executes_structured_copy_path_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    source_file = case_dir / "one.txt"
    create_text_file(source_file, "payload")
    target_dir = case_dir / "target"

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "copy file",
            "tool_name": "copy-path",
            "tool_arguments": {"source_path": str(source_file), "target_dir": str(target_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    tool_call = data["tool_calls"][0]
    assert data["final_state"] == "completed"
    assert tool_call["name"] == "copy-path"
    assert tool_call["result"]["copied_type"] == "file"
    assert (target_dir / "one.txt").read_text(encoding="utf-8") == "payload"
    assert source_file.exists()


def test_agent_executes_structured_move_path_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    source_dir = case_dir / "folder_a"
    create_text_file(source_dir / "nested" / "k.txt", "v")
    target_dir = case_dir / "target_root"

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "move folder",
            "tool_name": "move-path",
            "tool_arguments": {"source_path": str(source_dir), "target_dir": str(target_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    tool_call = data["tool_calls"][0]
    assert data["final_state"] == "completed"
    assert tool_call["name"] == "move-path"
    assert tool_call["result"]["moved_type"] == "directory"
    assert (target_dir / "folder_a" / "nested" / "k.txt").read_text(encoding="utf-8") == "v"
    assert not source_dir.exists()


def test_agent_executes_structured_reset_yolo_label_index_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "reset_dataset"
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text("3 0.1 0.2 0.3 0.4\n0 0.2 0.3 0.4 0.5\n", encoding="utf-8")

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "reset labels",
            "tool_name": "reset-yolo-label-index",
            "tool_arguments": {"input_dir": str(dataset_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    tool_call = data["tool_calls"][0]
    assert data["final_state"] == "completed"
    assert tool_call["name"] == "reset-yolo-label-index"
    assert tool_call["result"]["changed_lines"] == 1
    assert (labels_dir / "a.txt").read_text(encoding="utf-8").startswith("0 ")


def test_agent_executes_structured_voc_bar_crop_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "voc_bar_dataset"
    create_image(dataset_dir / "images" / "wide.jpg", color=(200, 200, 200), size=(200, 100))
    create_voc_xml(
        dataset_dir / "xmls" / "wide.xml",
        filename="wide.jpg",
        size=(200, 100),
        objects=[("bar", (40, 30, 160, 45))],
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "crop voc bars",
            "tool_name": "voc-bar-crop",
            "tool_arguments": {"input_dir": str(dataset_dir), "recursive": False},
        },
    )

    assert response.status_code == 200
    data = response.json()
    tool_call = data["tool_calls"][0]
    assert data["final_state"] == "completed"
    assert tool_call["name"] == "voc-bar-crop"
    assert tool_call["arguments"]["output_dir"] == f"{dataset_dir}_voc-bar-crop"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["generated_crops"] == 1


def test_agent_executes_structured_restore_voc_crops_batch_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    original_dir = case_dir / "original"
    create_image(original_dir / "images" / "wide.jpg", color=(200, 200, 200), size=(100, 100))
    create_voc_xml(
        original_dir / "xmls" / "wide.xml",
        filename="wide.jpg",
        size=(100, 100),
        objects=[("bar", (5, 5, 20, 20))],
    )
    edited_dir = case_dir / "edited"
    create_image(
        edited_dir / "images" / "wide_cx25_cy25_S50.jpg",
        color=(255, 0, 0),
        size=(50, 50),
    )
    create_voc_xml(
        edited_dir / "xmls" / "wide_cx25_cy25_S50.xml",
        filename="wide_cx25_cy25_S50.jpg",
        size=(50, 50),
        objects=[("bar", (10, 10, 30, 30))],
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "restore voc crops",
            "tool_name": "restore-voc-crops-batch",
            "tool_arguments": {
                "original_images_dir": str(original_dir / "images"),
                "original_xmls_dir": str(original_dir / "xmls"),
                "edited_crops_images_dir": str(edited_dir / "images"),
                "edited_crops_xmls_dir": str(edited_dir / "xmls"),
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    tool_call = data["tool_calls"][0]
    assert data["final_state"] == "completed"
    assert tool_call["name"] == "restore-voc-crops-batch"
    assert tool_call["arguments"]["output_dir"] == f"{edited_dir / 'images'}_restored"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["originals_processed"] == 1
    assert (edited_dir / "images_restored" / "images" / "wide.jpg").exists()
    assert (edited_dir / "images_restored" / "xmls" / "wide.xml").exists()


def test_agent_executes_structured_build_yolo_yaml_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "yaml_dataset"
    create_image(dataset_dir / "train" / "images" / "a.jpg", color=(10, 20, 30), size=(16, 16))
    create_image(dataset_dir / "val" / "images" / "b.jpg", color=(20, 30, 40), size=(16, 16))
    (dataset_dir / "classes.txt").write_text("object\n", encoding="utf-8")

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "build yaml",
            "tool_name": "build-yolo-yaml",
            "tool_arguments": {"input_dir": str(dataset_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "build-yolo-yaml"
    assert tool_call["arguments"]["output_yaml_path"] == str(dataset_dir / "data.yaml")
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["splits_included"] == ["train", "val"]
    assert (dataset_dir / "data.yaml").exists()


def test_agent_executes_structured_publish_incremental_yolo_dataset_tool(
    client,
    case_dir,
    monkeypatch,
    isolated_runtime,
) -> None:
    from app.schemas.preprocess import PublishYoloDatasetResponse

    dataset_dir = case_dir / "dataset_incremental"
    dataset_aug_dir = case_dir / "dataset_incremental_aug"
    create_yolo_dataset(dataset_dir, sample_count=1)
    create_yolo_dataset(dataset_aug_dir, sample_count=1)

    def fake_run_publish_incremental_yolo_dataset(_payload):
        return PublishYoloDatasetResponse(
            publish_mode="remote_sftp",
            output_yaml_path="/remote/datasets/demo_20260428/demo_20260428.yaml",
            dataset_root=str(dataset_dir),
            source_dataset_roots=[str(dataset_dir), str(dataset_aug_dir)],
            splits_included=["train"],
            classes_count=1,
            dataset_version="demo_20260428",
            published_dataset_dir="/remote/datasets/demo_20260428",
            staging_published_dataset_dir=str(case_dir / "staging"),
            staging_output_yaml_path=str(case_dir / "staging" / "demo_20260428.yaml"),
            local_archive_path=str(case_dir / "staging" / "demo_20260428.zip"),
            remote_target_host="172.31.1.42",
            remote_target_port=22,
            remote_archive_path="/remote/datasets/demo_20260428.zip",
            recommended_train_project="/remote/runs/detect",
            recommended_train_name="demo_20260428",
            last_yaml_merged=True,
            last_yaml_source="sftp",
        )

    monkeypatch.setattr(
        "app.agent.tools.registry._run_publish_incremental_yolo_dataset",
        fake_run_publish_incremental_yolo_dataset,
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "publish incremental dataset",
            "tool_name": "publish-incremental-yolo-dataset",
            "tool_arguments": {
                "input_dir": str(dataset_dir),
                "last_yaml": "sftp://172.31.1.42/remote/datasets/demo_prev/demo_prev.yaml",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "publish-incremental-yolo-dataset"
    assert tool_call["arguments"]["local_paths"] == [str(dataset_dir)]
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["published_dataset_dir"] == "/remote/datasets/demo_20260428"


def test_agent_executes_structured_publish_yolo_dataset_tool(
    client,
    case_dir,
    monkeypatch,
    isolated_runtime,
) -> None:
    from app.schemas.preprocess import PublishYoloDatasetResponse

    dataset_dir = case_dir / "dataset_publish_new"
    dataset_extra_dir = case_dir / "dataset_publish_new_extra"
    create_yolo_dataset(dataset_dir, sample_count=1)
    create_yolo_dataset(dataset_extra_dir, sample_count=1)

    def fake_run_publish_yolo_dataset(_payload):
        return PublishYoloDatasetResponse(
            publish_mode="remote_sftp",
            output_yaml_path="/remote/workspace/demo/demo_20260430/demo_20260430.yaml",
            dataset_root=str(dataset_dir),
            source_dataset_roots=[str(dataset_dir), str(dataset_extra_dir)],
            splits_included=["train"],
            classes_count=2,
            dataset_version="demo_20260430",
            published_dataset_dir="/remote/workspace/demo/datasets/demo_20260430",
            staging_published_dataset_dir=str(case_dir / "staging_publish"),
            staging_output_yaml_path=str(case_dir / "staging_publish" / "demo_20260430.yaml"),
            local_archive_path=str(case_dir / "staging_publish" / "demo_20260430.zip"),
            remote_target_host="172.31.1.42",
            remote_target_port=22,
            remote_archive_path="/remote/workspace/demo/datasets/demo_20260430.zip",
            recommended_train_project="/remote/workspace/demo/runs/detect",
            recommended_train_name="demo_20260430",
            last_yaml_merged=False,
            last_yaml_source=None,
        )

    monkeypatch.setattr(
        "app.agent.tools.registry._run_publish_yolo_dataset",
        fake_run_publish_yolo_dataset,
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "publish new dataset",
            "tool_name": "publish-yolo-dataset",
            "tool_arguments": {
                "local_paths": [str(dataset_dir), str(dataset_extra_dir)],
                "remote_target": "sftp://172.31.1.42/remote/workspace/demo",
                "classes": ["0", "1"],
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "publish-yolo-dataset"
    assert tool_call["arguments"]["input_dir"] == str(dataset_dir)
    assert tool_call["arguments"]["input_dirs"] == [str(dataset_extra_dir)]
    assert tool_call["arguments"]["publish_mode"] == "remote_sftp"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["published_dataset_dir"] == "/remote/workspace/demo/datasets/demo_20260430"


def test_agent_routes_chinese_zip_request(client, case_dir, monkeypatch, isolated_runtime) -> None:
    from app.agent import runtime as runtime_module

    source_dir = case_dir / "source_pack_cn"
    create_text_file(source_dir / "a.txt", "hello")
    monkeypatch.setattr(
        runtime_module,
        "request_tool_decision",
        lambda **_: LLMToolDecision(
            action="execute",
            tool_name="zip-folder",
            tool_arguments={"input_dir": str(source_dir)},
        ),
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"把这个目录打包成zip {source_dir}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "zip-folder"
    assert data["tool_calls"][0]["result"]["packed_files"] == 1


def test_agent_api_requires_auth_when_enabled(client, monkeypatch, isolated_runtime) -> None:
    monkeypatch.setenv("SELF_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("SELF_API_AUTH_ADMIN_PASSWORD", "secret-pass")
    monkeypatch.setenv("SELF_API_AUTH_SECRET_KEY", "unit-test-secret")
    get_settings.cache_clear()

    response = client.get("/api/v1/agent/tools")

    assert response.status_code == 401
