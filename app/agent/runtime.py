from dataclasses import dataclass

from app.agent.providers import ProviderCallError, request_tool_decision, select_provider
from app.agent.sessions import SQLiteAgentSessionStore, agent_session_store
from app.agent.tools.catalog import get_executable_tool_specs
from app.agent.tools.executor import AgentToolError, execute_tool
from app.agent.types import AgentRunRecord, LLMToolDecision, ToolCallRecord
from app.core.config import get_settings
from app.schemas.agent import AgentChatRequest


@dataclass(frozen=True)
class SessionResourceContext:
    dataset_root: str | None = None
    labels_dir: str | None = None
    images_dir: str | None = None


class AgentRuntime:
    def __init__(self, store: SQLiteAgentSessionStore | None = None) -> None:
        self._store = store or agent_session_store

    def chat(self, payload: AgentChatRequest) -> AgentRunRecord:
        settings = get_settings()
        session_id = payload.session_id or self._store.create_session_id()
        run_id = self._store.create_run_id()
        prior_runs = self._store.list_session_runs(session_id)
        provider = select_provider(
            settings,
            provider=payload.provider,
            model=payload.model,
        )
        decision, decision_error = self._decide_action(
            payload=payload,
            provider=provider,
            settings=settings,
            prior_runs=prior_runs,
        )
        if decision is not None and decision.action == "execute" and decision.tool_name:
            tool_name = decision.tool_name
            tool_arguments = self._hydrate_tool_arguments(
                tool_name,
                dict(decision.tool_arguments),
                prior_runs,
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
            message = tool_call.error or self._summarize_tool_result(tool_call.result)
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

    def _decide_action(
        self,
        *,
        payload: AgentChatRequest,
        provider,
        settings,
        prior_runs: list[AgentRunRecord],
    ) -> tuple[LLMToolDecision | None, str | None]:
        if payload.tool_name:
            return (
                LLMToolDecision(
                    action="execute",
                    tool_name=payload.tool_name,
                    tool_arguments=self._hydrate_tool_arguments(
                        payload.tool_name,
                        dict(payload.tool_arguments),
                        prior_runs,
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
                    conversation_context=self._build_conversation_context(prior_runs),
                ),
                None,
            )
        except ProviderCallError as exc:
            return None, str(exc)

    def _hydrate_tool_arguments(
        self,
        tool_name: str,
        arguments: dict,
        prior_runs: list[AgentRunRecord],
    ) -> dict:
        normalized = dict(arguments)
        if tool_name in {"move-path", "copy-path", "unzip-archive", "restore-voc-crops-batch"}:
            return normalized
        if normalized.get("input_dir"):
            return normalized
        resources = self._build_session_resource_context(prior_runs)
        inherited_path = self._resolve_input_dir_for_tool(tool_name, resources)
        if inherited_path:
            normalized["input_dir"] = inherited_path
        return normalized

    def _resolve_input_dir_for_tool(
        self,
        tool_name: str,
        resources: SessionResourceContext,
    ) -> str | None:
        prefer_labels_tools = {
            "scan-yolo-label-indices",
            "rewrite-yolo-label-indices",
            "reset-yolo-label-index",
        }
        prefer_dataset_tools = {
            "xml-to-yolo",
            "split-yolo-dataset",
            "yolo-sliding-window-crop",
            "yolo-augment",
            "annotate-visualize",
            "build-yolo-yaml",
            "aggregate-nested-dataset",
            "clean-nested-dataset-flat",
            "zip-folder",
            "publish-incremental-yolo-dataset",
            "publish-yolo-dataset",
            "voc-bar-crop",
            "discover-leaf-dirs",
        }
        if tool_name in prefer_labels_tools:
            return resources.labels_dir or resources.dataset_root
        if tool_name in prefer_dataset_tools:
            return resources.dataset_root or resources.labels_dir
        return resources.dataset_root or resources.labels_dir

    def _build_session_resource_context(
        self,
        prior_runs: list[AgentRunRecord],
    ) -> SessionResourceContext:
        dataset_root: str | None = None
        labels_dir: str | None = None
        images_dir: str | None = None
        for run in reversed(prior_runs):
            for tool_call in reversed(run.tool_calls):
                resource = self._extract_tool_resources(tool_call)
                dataset_root = dataset_root or resource.dataset_root
                labels_dir = labels_dir or resource.labels_dir
                images_dir = images_dir or resource.images_dir
                if dataset_root and labels_dir and images_dir:
                    return SessionResourceContext(
                        dataset_root=dataset_root,
                        labels_dir=labels_dir,
                        images_dir=images_dir,
                    )
        return SessionResourceContext(
            dataset_root=dataset_root,
            labels_dir=labels_dir,
            images_dir=images_dir,
        )

    def _extract_tool_resources(self, tool_call: ToolCallRecord) -> SessionResourceContext:
        result = tool_call.result if isinstance(tool_call.result, dict) else {}
        task_result = result.get("result") if isinstance(result.get("result"), dict) else {}
        raw_input_dir = self._first_path_value(
            task_result,
            result,
            tool_call.arguments,
            keys=("input_dir", "dataset_root", "source_dataset_root"),
        )
        dataset_root = raw_input_dir
        labels_dir = self._first_path_value(
            task_result,
            result,
            tool_call.arguments,
            keys=("labels_dir",),
        )
        images_dir = self._first_path_value(
            task_result,
            result,
            tool_call.arguments,
            keys=("images_dir",),
        )
        if raw_input_dir and raw_input_dir.rstrip("/").endswith("/labels"):
            labels_dir = labels_dir or raw_input_dir
            dataset_root = self._parent_dataset_root(raw_input_dir)
        elif raw_input_dir and raw_input_dir.rstrip("/").endswith("/images"):
            images_dir = images_dir or raw_input_dir
            dataset_root = self._parent_dataset_root(raw_input_dir)
        if labels_dir and not dataset_root:
            dataset_root = self._parent_dataset_root(labels_dir)
        if dataset_root and not labels_dir:
            labels_dir = self._join_child_path(dataset_root, "labels")
        if dataset_root and not images_dir:
            images_dir = self._join_child_path(dataset_root, "images")
        return SessionResourceContext(
            dataset_root=dataset_root,
            labels_dir=labels_dir,
            images_dir=images_dir,
        )

    def _first_path_value(self, *sources: dict, keys: tuple[str, ...]) -> str | None:
        for source in sources:
            for key in keys:
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return None

    def _parent_dataset_root(self, path: str) -> str | None:
        stripped = path.rstrip("/")
        if stripped.endswith("/labels"):
            return stripped[: -len("/labels")] or None
        if stripped.endswith("/images"):
            return stripped[: -len("/images")] or None
        return None

    def _join_child_path(self, root: str, child: str) -> str:
        return f"{root.rstrip('/')}/{child}"

    def _build_conversation_context(self, prior_runs: list[AgentRunRecord]) -> str | None:
        resources = self._build_session_resource_context(prior_runs)
        if not resources.dataset_root and not resources.labels_dir and not resources.images_dir:
            return None
        lines = ["Recent session resources:"]
        if resources.dataset_root:
            lines.append(f"- dataset_root: {resources.dataset_root}")
        if resources.labels_dir:
            lines.append(f"- labels_dir: {resources.labels_dir}")
        if resources.images_dir:
            lines.append(f"- images_dir: {resources.images_dir}")
        lines.append(
            "If the current request omits a path, choose the resource type that best fits the selected tool."
        )
        return "\n".join(lines)

    def _summarize_tool_result(self, result: dict | None) -> str:
        if not result:
            return "Tool executed."
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


agent_runtime = AgentRuntime()
