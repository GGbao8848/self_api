from pathlib import PurePosixPath
from urllib.parse import urlparse

from app.agent.providers import ProviderCallError, request_tool_decision, select_provider
from app.agent.sessions import SQLiteAgentSessionStore, agent_session_store
from app.agent.tools.catalog import get_executable_tool_specs
from app.agent.tools.executor import AgentToolError, execute_tool
from app.agent.types import AgentRunRecord, LLMToolDecision, ToolCallRecord
from app.core.config import get_settings
from app.schemas.agent import AgentChatRequest, AgentSessionSummaryResponse


class AgentRuntime:
    def __init__(self, store: SQLiteAgentSessionStore | None = None) -> None:
        self._store = store or agent_session_store

    def chat(self, payload: AgentChatRequest) -> AgentRunRecord:
        settings = get_settings()
        session_id = payload.session_id or self._store.create_session_id()
        run_id = self._store.create_run_id()
        session_runs = self._store.list_session_runs(session_id)
        session_publish_context = self._extract_publish_context(session_runs)
        session_summary = self._build_session_summary(session_publish_context)
        session_summary_message = self._build_session_summary_message(payload.message, session_summary)
        if session_summary_message is not None and not payload.tool_name:
            run = AgentRunRecord(
                session_id=session_id,
                run_id=run_id,
                user_message=payload.message,
                message=session_summary_message,
                final_state="completed",
                provider=None,
                model=None,
            )
            self._store.save_run(run)
            return run
        provider = select_provider(
            settings,
            provider=payload.provider,
            model=payload.model,
        )
        decision, decision_error = self._decide_action(
            payload=payload,
            provider=provider,
            settings=settings,
            session_publish_context=session_publish_context,
        )
        if decision is not None and decision.action == "execute" and decision.tool_name:
            tool_name = decision.tool_name
            tool_arguments = self._apply_session_publish_context(
                tool_name,
                dict(decision.tool_arguments),
                session_publish_context,
            )
            try:
                tool_call = execute_tool(tool_name, tool_arguments)
            except AgentToolError as exc:
                run = AgentRunRecord(
                    session_id=session_id,
                    run_id=run_id,
                    user_message=payload.message,
                    message=str(exc),
                    final_state="failed",
                    provider=provider.provider or None,
                    model=provider.model,
                    tool_calls=[],
                )
                self._store.save_run(run)
                return run

            state = "failed" if tool_call.error else "completed"
            message = tool_call.error or self._summarize_tool_result(
                tool_call=tool_call,
                session_publish_context=session_publish_context,
            )
            run = AgentRunRecord(
                session_id=session_id,
                run_id=run_id,
                user_message=payload.message,
                message=message,
                final_state=state,
                provider=provider.provider or None,
                model=provider.model,
                tool_calls=[tool_call],
            )
            self._store.save_run(run)
            return run

        if not provider.configured:
            run = AgentRunRecord(
                session_id=session_id,
                run_id=run_id,
                user_message=payload.message,
                message=provider.reason or "LLM provider is not configured",
                final_state="requires_provider",
                provider=provider.provider or None,
                model=provider.model,
            )
            self._store.save_run(run)
            return run

        if decision_error is not None:
            run = AgentRunRecord(
                session_id=session_id,
                run_id=run_id,
                user_message=payload.message,
                message=decision_error,
                final_state="failed",
                provider=provider.provider,
                model=provider.model,
            )
            self._store.save_run(run)
            return run

        if decision is not None and decision.action == "respond":
            run = AgentRunRecord(
                session_id=session_id,
                run_id=run_id,
                user_message=payload.message,
                message=decision.message or "No tool execution is needed.",
                final_state="completed",
                provider=provider.provider,
                model=provider.model,
            )
            self._store.save_run(run)
            return run

        if decision is not None and decision.action == "clarify":
            run = AgentRunRecord(
                session_id=session_id,
                run_id=run_id,
                user_message=payload.message,
                message=decision.message or "I need more details to choose a tool.",
                final_state="clarification_required",
                provider=provider.provider,
                model=provider.model,
            )
            self._store.save_run(run)
            return run

        run = AgentRunRecord(
            session_id=session_id,
            run_id=run_id,
            user_message=payload.message,
            message="I need more details to choose a tool.",
            final_state="clarification_required",
            provider=provider.provider,
            model=provider.model,
        )
        self._store.save_run(run)
        return run

    def get_session_summary(self, session_id: str) -> AgentSessionSummaryResponse:
        runs = self._store.list_session_runs(session_id)
        publish_context = self._extract_publish_context(runs)
        return self._build_session_summary(publish_context)

    def _decide_action(
        self,
        *,
        payload: AgentChatRequest,
        provider,
        settings,
        session_publish_context: dict,
    ) -> tuple[LLMToolDecision | None, str | None]:
        if payload.tool_name:
            return (
                LLMToolDecision(
                    action="execute",
                    tool_name=payload.tool_name,
                    tool_arguments=self._apply_session_publish_context(
                        payload.tool_name,
                        dict(payload.tool_arguments),
                        session_publish_context,
                    ),
                ),
                None,
            )
        if not provider.configured:
            return None, None

        try:
            return (
                request_tool_decision(
                    settings=settings,
                    provider=provider,
                    message=payload.message,
                    tools=get_executable_tool_specs(),
                    conversation_context=self._build_conversation_context(session_publish_context),
                ),
                None,
            )
        except ProviderCallError as exc:
            return None, str(exc)

    def _summarize_tool_result(
        self,
        *,
        tool_call: ToolCallRecord,
        session_publish_context: dict,
    ) -> str:
        result = tool_call.result
        if not result:
            return "Tool executed."
        effective_result = self._extract_effective_tool_result(result) if isinstance(result, dict) else None
        if tool_call.name == "check-latest-dataset-version" and isinstance(effective_result, dict):
            latest_yaml = effective_result.get("latest_yaml")
            if isinstance(latest_yaml, str) and latest_yaml.strip():
                prior_latest_yaml = session_publish_context.get("last_yaml")
                if isinstance(prior_latest_yaml, str) and prior_latest_yaml.strip():
                    if latest_yaml.strip() == prior_latest_yaml.strip():
                        return f"智能体最近一次提交的就是当前最新版本，路径为: {latest_yaml}"
                    return (
                        f"存在后续人工/外部提交的更新版本，路径为: {latest_yaml}。"
                        f"智能体最近一次提交路径为: {prior_latest_yaml}"
                    )
                return f"当前最新版本路径为: {latest_yaml}"
        if "task_id" in result:
            task_result = result.get("result")
            output_dir = task_result.get("output_dir") if isinstance(task_result, dict) else None
            labels_dir = task_result.get("labels_dir") if isinstance(task_result, dict) else None
            output_path = output_dir or labels_dir
            suffix = f"; output={output_path}" if output_path else ""
            return (
                f"Task {result.get('task_id')} finished with state="
                f"{result.get('state')}{suffix}."
            )
        if "indices" in result:
            indices = result.get("indices") or []
            return (
                f"Found {result.get('total_objects', 0)} objects in "
                f"{result.get('total_label_files', 0)} label files; "
                f"indices={indices}."
            )
        if "changed_lines" in result:
            return (
                f"Rewrote {result.get('changed_lines', 0)} label lines in "
                f"{result.get('modified_label_files', 0)} files."
            )
        return "Tool executed."

    def _build_conversation_context(self, publish_context: dict) -> str | None:
        if not publish_context:
            return None
        lines = [
            "Conversation context from the current session:",
            "- Reuse these only when they match the user's current intent.",
        ]
        if publish_context.get("detector_name"):
            lines.append(f"- last_detector_name: {publish_context['detector_name']}")
        if publish_context.get("remote_target"):
            lines.append(f"- last_remote_target: {publish_context['remote_target']}")
        if publish_context.get("last_yaml"):
            lines.append(f"- last_successful_yaml: {publish_context['last_yaml']}")
        if publish_context.get("classes"):
            lines.append(f"- last_classes: {publish_context['classes']}")
        if publish_context.get("use_index_as_class_names"):
            lines.append("- last_publish_used_numeric_class_names: true")
        return "\n".join(lines)

    def _build_session_summary(self, publish_context: dict) -> AgentSessionSummaryResponse:
        if not publish_context:
            return AgentSessionSummaryResponse(has_context=False)
        latest_yaml = self._resolve_live_last_yaml(publish_context) or publish_context.get("last_yaml")
        detector_name = publish_context.get("detector_name") or self._extract_detector_name(
            latest_yaml,
            publish_context.get("remote_target"),
        )
        dataset_version = self._extract_dataset_version_from_yaml(latest_yaml)
        classes = publish_context.get("classes")
        return AgentSessionSummaryResponse(
            has_context=bool(detector_name or latest_yaml or publish_context.get("remote_target")),
            detector_name=detector_name if isinstance(detector_name, str) and detector_name.strip() else None,
            dataset_version=dataset_version,
            latest_yaml=latest_yaml if isinstance(latest_yaml, str) and latest_yaml.strip() else None,
            remote_target=publish_context.get("remote_target"),
            classes=classes if isinstance(classes, list) else [],
            use_index_as_class_names=publish_context.get("use_index_as_class_names") is True,
        )

    def _build_session_summary_message(
        self,
        message: str,
        summary: AgentSessionSummaryResponse,
    ) -> str | None:
        compact = "".join(message.lower().split())
        triggers = (
            "当前维护的是哪个模型",
            "当前维护哪个模型",
            "当前模型",
            "当前维护对象",
            "当前最新yaml是哪个",
            "当前最新yaml",
            "最新yaml是哪个",
            "最新yaml",
            "会话摘要",
            "当前摘要",
            "小结视图",
            "小结",
        )
        if not any(trigger in compact for trigger in triggers):
            return None
        if not summary.has_context:
            return "当前会话里还没有维护对象记录，先执行一次发布或最新版本查询。"

        lines = ["当前会话维护摘要"]
        if summary.detector_name:
            lines.append(f"- 模型: {summary.detector_name}")
        if summary.dataset_version:
            lines.append(f"- 最新版本: {summary.dataset_version}")
        if summary.latest_yaml:
            lines.append(f"- 最新 yaml: {summary.latest_yaml}")
        if summary.remote_target:
            lines.append(f"- 远端目标: {summary.remote_target}")
        if summary.classes:
            lines.append(f"- 类别: {', '.join(summary.classes)}")
        elif summary.use_index_as_class_names:
            lines.append("- 类别: 使用数字索引作为类名")
        return "\n".join(lines)

    def _apply_session_publish_context(
        self,
        tool_name: str | None,
        tool_arguments: dict,
        publish_context: dict,
    ) -> dict:
        if not tool_name or not publish_context:
            return tool_arguments
        normalized = dict(tool_arguments)
        if tool_name == "check-latest-dataset-version":
            if self._is_missing_like_value(normalized.get("remote_target")) and publish_context.get("remote_target"):
                normalized["remote_target"] = publish_context["remote_target"]
            return normalized
        if tool_name == "publish-incremental-yolo-dataset":
            live_last_yaml = self._resolve_live_last_yaml(publish_context)
            if live_last_yaml:
                if self._is_missing_like_value(normalized.get("last_yaml")):
                    normalized["last_yaml"] = live_last_yaml
            elif publish_context.get("last_yaml"):
                if self._is_missing_like_value(normalized.get("last_yaml")):
                    normalized["last_yaml"] = publish_context["last_yaml"]
            return normalized
        if tool_name == "publish-yolo-dataset":
            for key in ("remote_target", "detector_name", "classes"):
                if publish_context.get(key) and self._is_missing_like_value(normalized.get(key)):
                    normalized[key] = publish_context[key]
            if publish_context.get("use_index_as_class_names"):
                normalized.setdefault("use_index_as_class_names", True)
            return normalized
        return normalized

    def _extract_publish_context(self, runs: list[AgentRunRecord]) -> dict:
        for run in reversed(runs):
            for tool_call in reversed(run.tool_calls):
                context = self._extract_publish_context_from_tool_call(tool_call)
                if context:
                    return context
        return {}

    def _extract_publish_context_from_tool_call(self, tool_call: ToolCallRecord) -> dict:
        if tool_call.error:
            return {}
        if tool_call.name not in {"publish-yolo-dataset", "publish-incremental-yolo-dataset", "check-latest-dataset-version"}:
            return {}
        raw_result = tool_call.result or {}
        if not isinstance(raw_result, dict):
            return {}
        result = self._extract_effective_tool_result(raw_result)
        if not isinstance(result, dict):
            return {}
        if tool_call.name != "check-latest-dataset-version" and result.get("publish_mode") != "remote_sftp":
            return {}

        arguments = tool_call.arguments or {}
        context: dict = {}
        detector_name = arguments.get("detector_name")
        if isinstance(detector_name, str) and detector_name.strip():
            context["detector_name"] = detector_name.strip()
        elif tool_call.name == "check-latest-dataset-version":
            candidate = result.get("detector_name")
            if isinstance(candidate, str) and candidate.strip():
                context["detector_name"] = candidate.strip()

        remote_target = arguments.get("remote_target")
        if isinstance(remote_target, str) and remote_target.strip():
            context["remote_target"] = remote_target.strip().rstrip("/")
        else:
            inferred_remote_target = self._infer_remote_target_from_result(result)
            if inferred_remote_target:
                context["remote_target"] = inferred_remote_target

        last_yaml = None
        if tool_call.name == "check-latest-dataset-version":
            candidate = result.get("latest_yaml")
            if isinstance(candidate, str) and candidate.strip():
                last_yaml = candidate.strip()
        output_yaml_path = result.get("output_yaml_path")
        if not last_yaml and isinstance(output_yaml_path, str) and output_yaml_path.strip():
            host = result.get("remote_target_host")
            port = result.get("remote_target_port")
            last_yaml = self._to_sftp_uri(output_yaml_path.strip(), host, port)
        if not last_yaml and tool_call.name == "publish-incremental-yolo-dataset":
            candidate = arguments.get("last_yaml")
            if isinstance(candidate, str) and candidate.strip():
                last_yaml = candidate.strip()
        if last_yaml:
            context["last_yaml"] = last_yaml

        classes = arguments.get("classes")
        if isinstance(classes, list):
            cleaned = [item.strip() for item in classes if isinstance(item, str) and item.strip()]
            if cleaned:
                context["classes"] = cleaned
        if arguments.get("use_index_as_class_names") is True:
            context["use_index_as_class_names"] = True
        return context

    def _extract_effective_tool_result(self, raw_result: dict) -> dict | None:
        nested = raw_result.get("result")
        if isinstance(nested, dict):
            return nested
        return raw_result

    def _is_missing_like_value(self, value) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"", "null", "none"}
        return False

    def _infer_remote_target_from_result(self, result: dict) -> str | None:
        output_yaml_path = result.get("output_yaml_path")
        host = result.get("remote_target_host")
        port = result.get("remote_target_port")
        if not isinstance(output_yaml_path, str) or not output_yaml_path.strip():
            return None
        path = PurePosixPath(urlparse(output_yaml_path).path if output_yaml_path.startswith(("sftp://", "ssh://")) else output_yaml_path)
        parts = path.parts
        bucket_idx = next((i for i, part in enumerate(parts) if part in {"dataset", "datasets"}), None)
        if bucket_idx is None or bucket_idx < 1:
            return None
        detector_path = PurePosixPath(*parts[:bucket_idx])
        return self._to_sftp_uri(detector_path.as_posix(), host, port)

    def _to_sftp_uri(self, path: str, host, port) -> str | None:
        if not isinstance(host, str) or not host.strip():
            return None
        port_num = port if isinstance(port, int) and port > 0 else 22
        if port_num == 22:
            return f"sftp://{host}{path}"
        return f"sftp://{host}:{port_num}{path}"

    def _resolve_live_last_yaml(self, publish_context: dict) -> str | None:
        remote_target = publish_context.get("remote_target")
        if not isinstance(remote_target, str) or not remote_target.strip():
            return None
        settings = get_settings()
        if not settings.remote_sftp_username or not settings.remote_sftp_private_key_path:
            return None
        try:
            from app.services.remote_transfer import find_latest_remote_yaml

            return find_latest_remote_yaml(
                remote_target,
                username=settings.remote_sftp_username,
                private_key_path=settings.remote_sftp_private_key_path,
                port=settings.remote_sftp_port,
            )
        except ValueError:
            return None

    def _extract_dataset_version_from_yaml(self, yaml_path: str | None) -> str | None:
        if not isinstance(yaml_path, str) or not yaml_path.strip():
            return None
        path = PurePosixPath(urlparse(yaml_path).path if yaml_path.startswith(("sftp://", "ssh://")) else yaml_path)
        parts = path.parts
        bucket_idx = next((i for i, part in enumerate(parts) if part in {"dataset", "datasets"}), None)
        if bucket_idx is None or bucket_idx + 1 >= len(parts):
            return None
        dataset_version = parts[bucket_idx + 1]
        return dataset_version or None

    def _extract_detector_name(self, yaml_path: str | None, remote_target: str | None) -> str | None:
        for candidate in (yaml_path, remote_target):
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            path = PurePosixPath(urlparse(candidate).path if candidate.startswith(("sftp://", "ssh://")) else candidate)
            parts = path.parts
            bucket_idx = next((i for i, part in enumerate(parts) if part in {"dataset", "datasets"}), None)
            if bucket_idx is not None and bucket_idx >= 1:
                detector_name = parts[bucket_idx - 1]
                if detector_name:
                    return detector_name
            if parts:
                tail = parts[-1]
                if tail:
                    return tail
        return None


agent_runtime = AgentRuntime()
