from app.agent.providers import ProviderCallError, request_tool_decision, select_provider
from app.agent.sessions import InMemoryAgentSessionStore, agent_session_store
from app.agent.tools.catalog import get_executable_tool_specs
from app.agent.tools.executor import AgentToolError, execute_tool
from app.agent.types import AgentRunRecord, LLMToolDecision
from app.core.config import get_settings
from app.schemas.agent import AgentChatRequest


class AgentRuntime:
    def __init__(self, store: InMemoryAgentSessionStore | None = None) -> None:
        self._store = store or agent_session_store

    def chat(self, payload: AgentChatRequest) -> AgentRunRecord:
        settings = get_settings()
        session_id = payload.session_id or self._store.create_session_id()
        run_id = self._store.create_run_id()
        provider = select_provider(
            settings,
            provider=payload.provider,
            model=payload.model,
        )
        decision, decision_error = self._decide_action(
            payload=payload,
            provider=provider,
            settings=settings,
        )
        if decision is not None and decision.action == "execute" and decision.tool_name:
            tool_name = decision.tool_name
            tool_arguments = dict(decision.tool_arguments)
            try:
                tool_call = execute_tool(tool_name, tool_arguments)
            except AgentToolError as exc:
                run = AgentRunRecord(
                    session_id=session_id,
                    run_id=run_id,
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
    ) -> tuple[LLMToolDecision | None, str | None]:
        if payload.tool_name:
            return (
                LLMToolDecision(
                    action="execute",
                    tool_name=payload.tool_name,
                    tool_arguments=dict(payload.tool_arguments),
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
                ),
                None,
            )
        except ProviderCallError as exc:
            return None, str(exc)

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
