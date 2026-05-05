import time
from dataclasses import dataclass
from threading import Thread

from pydantic import ValidationError

from app.agent.langgraph_agent import LangGraphAgentExecutor
from app.agent.providers import ProviderCallError, request_tool_decision, select_provider
from app.agent.langgraph_pipeline import matches_langgraph_pipeline_request, run_pipeline
from app.agent.sessions import SQLiteAgentSessionStore, agent_session_store
from app.agent.tools.catalog import get_executable_tool_specs
from app.agent.tools.registry import get_tool_definition
from app.agent.types import AgentRunRecord, AgentStepRecord, LLMToolDecision, ProviderSelection, ToolCallRecord
from app.core.config import get_settings
from app.schemas.agent import AgentChatRequest
from app.services.task_manager import cancel_task, get_task, submit_task


@dataclass(frozen=True)
class SessionResourceContext:
    dataset_root: str | None = None
    labels_dir: str | None = None
    images_dir: str | None = None


class AgentRunCancelledError(RuntimeError):
    pass


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class AgentRuntime:
    def __init__(self, store: SQLiteAgentSessionStore | None = None) -> None:
        self._store = store or agent_session_store

    def _settings(self):
        return get_settings()

    def _save_run(self, run: AgentRunRecord) -> None:
        self._store.save_run(run)

    def _now(self) -> str:
        return _now_iso()

    def chat(self, payload: AgentChatRequest) -> AgentRunRecord:
        if payload.async_run:
            return self._submit_long_run(payload)
        return self._run_inline(payload)

    def cancel_run(self, run_id: str) -> AgentRunRecord | None:
        return self._store.cancel_run(run_id)

    def retry_run(
        self,
        run_id: str,
        *,
        message: str | None = None,
        async_run: bool = True,
        max_steps: int | None = None,
    ) -> AgentRunRecord | None:
        original = self._store.get_run(run_id)
        if original is None:
            return None
        retry_message = (message or original.user_message or "").strip()
        if not retry_message:
            raise ValueError(f"agent run has no retryable message: {run_id}")
        return self._spawn_followup_run(
            original,
            message=retry_message,
            async_run=async_run,
            max_steps=max_steps,
            trigger_kind="retry",
        )

    def continue_run(
        self,
        run_id: str,
        *,
        message: str,
        async_run: bool = True,
        max_steps: int | None = None,
    ) -> AgentRunRecord | None:
        original = self._store.get_run(run_id)
        if original is None:
            return None
        next_message = (message or "").strip()
        if not next_message:
            raise ValueError("continue message must not be empty")
        return self._spawn_followup_run(
            original,
            message=next_message,
            async_run=async_run,
            max_steps=max_steps,
            trigger_kind="continue",
        )

    def resume_run(
        self,
        run_id: str,
        *,
        max_steps: int | None = None,
    ) -> AgentRunRecord | None:
        run = self._store.get_run(run_id)
        if run is None:
            return None
        if run.final_state != "interrupted":
            raise ValueError(f"agent run is not resumable: {run_id}")
        payload = self._payload_from_run(run, max_steps=max_steps)
        if not payload.async_run:
            raise ValueError("only async agent runs can be resumed")
        run.final_state = "accepted"
        run.message = "Agent run resume accepted."
        run.updated_at = _now_iso()
        run.finished_at = None
        run.checkpoint = {
            **run.checkpoint,
            "resume_required": False,
            "resume_requested_at": run.updated_at,
        }
        self._store.save_run(run)
        self._start_run_worker(run_id, payload, resume=True)
        return run

    def _spawn_followup_run(
        self,
        original: AgentRunRecord,
        *,
        message: str,
        async_run: bool,
        max_steps: int | None,
        trigger_kind: str,
    ) -> AgentRunRecord:
        payload = AgentChatRequest(
            session_id=original.session_id,
            provider=original.provider,
            model=original.model,
            message=message,
            async_run=async_run,
            max_steps=max_steps,
            parent_run_id=original.run_id,
            trigger_kind=trigger_kind,
        )
        return self.chat(payload)

    def _payload_from_run(self, run: AgentRunRecord, *, max_steps: int | None = None) -> AgentChatRequest:
        payload_data = dict(run.request_payload)
        if not payload_data:
            raise ValueError(f"agent run has no stored request payload: {run.run_id}")
        if max_steps is not None:
            payload_data["max_steps"] = max_steps
        payload_data.setdefault("session_id", run.session_id)
        payload_data.setdefault("provider", run.provider)
        payload_data.setdefault("model", run.model)
        return AgentChatRequest(**payload_data)

    def _start_run_worker(self, run_id: str, payload: AgentChatRequest, *, resume: bool = False) -> None:
        worker = Thread(
            target=self._execute_long_run,
            args=(run_id, payload),
            kwargs={"resume": resume},
            daemon=True,
            name=f"agent-run-{run_id[:8]}",
        )
        worker.start()

    def _run_inline(self, payload: AgentChatRequest) -> AgentRunRecord:
        settings = self._settings()
        session_id = payload.session_id or self._store.create_session_id()
        run_id = self._store.create_run_id()
        prior_runs = self._store.list_session_runs(session_id)
        parent_run = self._resolve_parent_run(payload, prior_runs)
        root_run_id = parent_run.root_run_id or parent_run.run_id if parent_run is not None else run_id
        provider = select_provider(
            settings,
            provider=payload.provider,
            model=payload.model,
        )
        now = _now_iso()
        if not payload.tool_name and matches_langgraph_pipeline_request(payload.message):
            provider = ProviderSelection(
                provider="langgraph",
                model="deterministic-preprocess-pipeline",
                configured=True,
            )
        run = AgentRunRecord(
            session_id=session_id,
            run_id=run_id,
            user_message=payload.message,
            message="Agent is planning the task.",
            final_state="running",
            parent_run_id=parent_run.run_id if parent_run is not None else payload.parent_run_id,
            root_run_id=root_run_id,
            trigger_kind=payload.trigger_kind,
            plan_summary=self._build_initial_plan_summary(payload, prior_runs),
            created_at=now,
            updated_at=now,
            provider=provider.provider or None,
            model=provider.model,
            request_payload=payload.model_dump(),
            checkpoint={
                "mode": "inline",
                "phase": "planning",
                "max_steps": payload.max_steps or settings.agent_max_steps,
                "executed_signatures": [],
            },
        )
        self._store.save_run(run)
        try:
            self._record_plan_step(run, payload, prior_runs, resume=False)
            LangGraphAgentExecutor(
                self,
                run=run,
                payload=payload,
                prior_runs=prior_runs,
                provider=provider,
                resume=False,
            ).execute()
            if run.final_state not in {"completed", "failed", "cancelled", "clarification_required", "requires_provider"}:
                self._finalize_run(
                    run,
                    state="failed",
                    message="langgraph inline execution finished without a terminal run state",
                )
        except AgentRunCancelledError:
            self._finalize_run(run, state="cancelled", message=f"agent run cancelled: {run.run_id}")
        except Exception as exc:  # noqa: BLE001
            self._finalize_run(run, state="failed", message=str(exc))
        return run

    def _submit_long_run(self, payload: AgentChatRequest) -> AgentRunRecord:
        settings = get_settings()
        session_id = payload.session_id or self._store.create_session_id()
        run_id = self._store.create_run_id()
        prior_runs = self._store.list_session_runs(session_id)
        parent_run = self._resolve_parent_run(payload, prior_runs)
        provider = select_provider(
            settings,
            provider=payload.provider,
            model=payload.model,
        )
        if not payload.tool_name and matches_langgraph_pipeline_request(payload.message):
            provider = ProviderSelection(
                provider="langgraph",
                model="deterministic-preprocess-pipeline",
                configured=True,
            )
        now = _now_iso()
        root_run_id = parent_run.root_run_id or parent_run.run_id if parent_run is not None else run_id
        run = AgentRunRecord(
            session_id=session_id,
            run_id=run_id,
            user_message=payload.message,
            message="Agent run accepted.",
            final_state="accepted",
            parent_run_id=parent_run.run_id if parent_run is not None else payload.parent_run_id,
            root_run_id=root_run_id,
            trigger_kind=payload.trigger_kind,
            plan_summary=self._build_initial_plan_summary(payload, prior_runs),
            created_at=now,
            updated_at=now,
            provider=provider.provider or None,
            model=provider.model,
            steps=[],
            request_payload=payload.model_dump(),
            checkpoint={
                "mode": "long_run",
                "phase": "accepted",
                "max_steps": payload.max_steps or settings.agent_max_steps,
                "executed_signatures": [],
            },
        )
        self._store.save_run(run)
        self._start_run_worker(run_id, payload)
        return run

    def _execute_long_run(self, run_id: str, payload: AgentChatRequest, *, resume: bool = False) -> None:
        run = self._store.get_run(run_id)
        if run is None:
            return
        settings = self._settings()
        prior_runs = self._store.list_session_runs(run.session_id)
        prior_runs = [item for item in prior_runs if item.run_id != run_id]
        provider = select_provider(
            settings,
            provider=payload.provider,
            model=payload.model,
        )
        run.provider = provider.provider or None
        run.model = provider.model
        run.final_state = "running"
        run.message = "Agent is resuming the task." if resume else "Agent is planning the task."
        run.updated_at = _now_iso()
        run.checkpoint = {
            **run.checkpoint,
            "mode": "long_run",
            "phase": "planning",
            "max_steps": payload.max_steps or settings.agent_max_steps,
            "resume_required": False,
        }
        self._store.save_run(run)

        try:
            self._record_plan_step(run, payload, prior_runs, resume=resume)
            LangGraphAgentExecutor(
                self,
                run=run,
                payload=payload,
                prior_runs=prior_runs,
                provider=provider,
                resume=resume,
            ).execute()
            if run.final_state not in {"completed", "failed", "cancelled", "clarification_required", "requires_provider"}:
                self._finalize_run(
                    run,
                    state="failed",
                    message="langgraph execution finished without a terminal run state",
                )
        except AgentRunCancelledError:
            self._finalize_run(
                run,
                state="cancelled",
                message=f"agent run cancelled: {run.run_id}",
            )
        except Exception as exc:  # noqa: BLE001
            self._finalize_run(run, state="failed", message=str(exc))

    def _execute_explicit_tool_step(
        self,
        run: AgentRunRecord,
        payload: AgentChatRequest,
        prior_runs: list[AgentRunRecord],
        *,
        resume: bool = False,
    ) -> None:
        tool_name = payload.tool_name or ""
        tool_arguments = self._hydrate_tool_arguments(
            tool_name,
            dict(payload.tool_arguments),
            prior_runs,
        )
        self._execute_tool_step(
            run,
            tool_name,
            tool_arguments,
            max_steps=payload.max_steps or get_settings().agent_max_steps,
            executed_signatures=set(),
            resume=resume,
        )
        if run.final_state in {"failed", "cancelled"}:
            return
        last_call = run.tool_calls[-1] if run.tool_calls else None
        self._finalize_run(
            run,
            state="completed",
            message=self._summarize_tool_result(last_call.result if last_call else None),
        )

    def _record_decision_step(
        self,
        run: AgentRunRecord,
        payload: AgentChatRequest,
        provider,
        settings,
        prior_runs: list[AgentRunRecord],
    ) -> LLMToolDecision | None:
        step = self._start_step(run, kind="decision", title="Plan next action")
        try:
            decision, decision_error = self._decide_action(
                payload=payload,
                provider=provider,
                settings=settings,
                prior_runs=prior_runs,
                current_run=run,
            )
            if decision_error is not None:
                step.status = "failed"
                step.message = decision_error
                step.finished_at = _now_iso()
                run.message = decision_error
                run.final_state = "failed"
                run.updated_at = _now_iso()
                self._store.save_run(run)
                return None
            if decision is None:
                step.status = "failed"
                step.message = "model returned no decision"
                step.finished_at = _now_iso()
                run.message = step.message
                run.final_state = "failed"
                run.updated_at = _now_iso()
                self._store.save_run(run)
                return None
            step.status = "completed"
            step.message = decision.message or (
                f"action={decision.action}" + (f", tool={decision.tool_name}" if decision.tool_name else "")
            )
            step.details = {
                "action": decision.action,
                "plan_summary": decision.plan_summary,
                "tool_name": decision.tool_name,
                "tool_arguments": decision.tool_arguments,
            }
            if decision.plan_summary:
                run.plan_summary = decision.plan_summary
            step.finished_at = _now_iso()
            run.message = step.message
            run.updated_at = _now_iso()
            self._store.save_run(run)
            return decision
        except Exception:
            step.status = "failed"
            step.finished_at = _now_iso()
            run.updated_at = _now_iso()
            self._store.save_run(run)
            raise

    def _execute_tool_step(
        self,
        run: AgentRunRecord,
        tool_name: str,
        tool_arguments: dict,
        *,
        max_steps: int,
        executed_signatures: set[tuple[str, tuple[tuple[str, str], ...]]],
        resume: bool = False,
    ) -> None:
        definition = get_tool_definition(tool_name)
        if definition is None:
            self._finalize_run(run, state="failed", message=f"tool is not executable yet: {tool_name}")
            return
        normalized_tool_arguments = self._normalize_tool_arguments(definition, tool_arguments)
        step = self._start_step(run, kind="tool", title=f"Execute {tool_name}", tool_name=tool_name)
        step.details = {
            "request": dict(normalized_tool_arguments),
            "resume_mode": resume,
        }
        run.checkpoint = {
            **run.checkpoint,
            "phase": "tool",
            "max_steps": max_steps,
            "executed_signatures": self._dump_executed_signatures(executed_signatures),
            "current_tool": {
                "tool_name": tool_name,
                "tool_arguments": dict(normalized_tool_arguments),
                "step_id": step.step_id,
            },
        }
        self._store.save_run(run)
        try:
            payload = definition.request_model(**normalized_tool_arguments)
        except (ValidationError, ValueError) as exc:
            tool_call = ToolCallRecord(name=tool_name, arguments=normalized_tool_arguments, error=str(exc))
            step.status = "failed"
            step.message = str(exc)
            step.tool_call = tool_call
            step.finished_at = _now_iso()
            run.tool_calls.append(tool_call)
            run.updated_at = _now_iso()
            self._store.save_run(run)
            self._finalize_run(run, state="failed", message=str(exc))
            return

        if definition.async_task:
            self._execute_async_tool_step(
                run,
                step,
                definition,
                payload,
                max_steps=max_steps,
                executed_signatures=executed_signatures,
            )
            return

        try:
            result = definition.runner(payload).model_dump()
            tool_call = ToolCallRecord(
                name=tool_name,
                arguments=payload.model_dump(),
                result=result,
            )
            step.status = "completed"
            step.message = self._summarize_tool_result(result)
            step.tool_call = tool_call
            step.details = {"result_keys": sorted(result.keys())}
            step.finished_at = _now_iso()
            run.tool_calls.append(tool_call)
            run.updated_at = _now_iso()
            run.message = step.message or run.message
            run.checkpoint = {
                **run.checkpoint,
                "phase": "decision",
                "executed_signatures": self._dump_executed_signatures(executed_signatures),
            }
            run.checkpoint.pop("current_tool", None)
            self._store.save_run(run)
        except Exception as exc:  # noqa: BLE001
            tool_call = ToolCallRecord(name=tool_name, arguments=payload.model_dump(), error=str(exc))
            step.status = "failed"
            step.message = str(exc)
            step.tool_call = tool_call
            step.finished_at = _now_iso()
            run.tool_calls.append(tool_call)
            run.updated_at = _now_iso()
            run.checkpoint.pop("current_tool", None)
            self._store.save_run(run)
            self._finalize_run(run, state="failed", message=str(exc))

    def _execute_async_tool_step(
        self,
        run,
        step,
        definition,
        payload,
        *,
        max_steps: int,
        executed_signatures: set[tuple[str, tuple[tuple[str, str], ...]]],
    ) -> None:
        tool_name = definition.name
        task_id = self._submit_async_task_for_definition(definition, payload)
        step.task_id = task_id
        step.task_type = definition.task_type or tool_name.replace("-", "_")
        step.message = self._format_async_submission_message(tool_name)
        step.details = {
            **step.details,
            "task_state": "pending",
            "task_id": task_id,
            "task_type": step.task_type,
        }
        run.final_state = "waiting_task"
        run.message = step.message
        run.updated_at = _now_iso()
        run.checkpoint = {
            **run.checkpoint,
            "phase": "waiting_task",
            "max_steps": max_steps,
            "executed_signatures": self._dump_executed_signatures(executed_signatures),
            "current_tool": {
                "tool_name": tool_name,
                "tool_arguments": payload.model_dump(),
                "task_id": task_id,
                "task_type": step.task_type,
                "step_id": step.step_id,
            },
        }
        self._store.save_run(run)

        self._wait_for_async_task(
            run,
            step,
            tool_name=tool_name,
            tool_arguments=payload.model_dump(),
            task_id=task_id,
            max_steps=max_steps,
            executed_signatures=executed_signatures,
        )

    def _submit_async_task_for_definition(self, definition, payload) -> str:
        return submit_task(
            task_type=definition.task_type or definition.name.replace("-", "_"),
            runner=lambda: definition.runner(payload).model_dump(),
        )

    def _wait_for_async_task(
        self,
        run: AgentRunRecord,
        step: AgentStepRecord,
        *,
        tool_name: str,
        tool_arguments: dict,
        task_id: str,
        max_steps: int,
        executed_signatures: set[tuple[str, tuple[tuple[str, str], ...]]],
    ) -> None:
        while True:
            self._ensure_run_active(run.run_id)
            task = get_task(task_id)
            if task is None:
                tool_call = ToolCallRecord(
                    name=tool_name,
                    arguments=tool_arguments,
                    error=f"task not found: {task_id}",
                )
                step.status = "failed"
                step.message = tool_call.error
                step.tool_call = tool_call
                step.finished_at = _now_iso()
                run.tool_calls.append(tool_call)
                run.updated_at = _now_iso()
                self._store.save_run(run)
                self._finalize_run(run, state="failed", message=tool_call.error or "task failed")
                return
            if task["state"] in {"pending", "running"}:
                run.final_state = "waiting_task"
                progress_text = self._format_task_progress(task)
                run.message = self._format_task_wait_message(
                    tool_name,
                    task,
                    progress_text=progress_text,
                )
                step.message = run.message
                step.details = {
                    **step.details,
                    **self._build_task_wait_details(task),
                }
                run.updated_at = _now_iso()
                self._store.save_run(run)
                time.sleep(0.2)
                continue
            if task["state"] == "interrupted":
                definition = get_tool_definition(tool_name)
                if definition is None:
                    self._finalize_run(run, state="failed", message=f"tool is not executable yet: {tool_name}")
                    return
                replacement_payload = definition.request_model(**tool_arguments)
                replacement_task_id = self._submit_async_task_for_definition(definition, replacement_payload)
                step.task_id = replacement_task_id
                step.task_type = definition.task_type or tool_name.replace("-", "_")
                step.status = "running"
                step.message = self._format_async_resubmission_message(tool_name)
                step.finished_at = None
                step.details = {
                    **step.details,
                    "resumed_from_task_id": task_id,
                    "task_id": replacement_task_id,
                    "task_type": step.task_type,
                    "task_state": "pending",
                }
                run.final_state = "waiting_task"
                run.message = step.message
                run.updated_at = _now_iso()
                run.checkpoint = {
                    **run.checkpoint,
                    "phase": "waiting_task",
                    "max_steps": max_steps,
                    "executed_signatures": self._dump_executed_signatures(executed_signatures),
                    "current_tool": {
                        "tool_name": tool_name,
                        "tool_arguments": dict(tool_arguments),
                        "task_id": replacement_task_id,
                        "task_type": step.task_type,
                        "step_id": step.step_id,
                    },
                }
                self._store.save_run(run)
                task_id = replacement_task_id
                continue

            compact_result = self._compact_task_result(task)
            tool_call = ToolCallRecord(
                name=tool_name,
                arguments=tool_arguments,
                result=compact_result,
                error=compact_result.get("error"),
            )
            step.tool_call = tool_call
            step.finished_at = _now_iso()
            run.tool_calls.append(tool_call)
            run.updated_at = _now_iso()
            if task["state"] == "succeeded":
                step.status = "completed"
                step.message = self._summarize_tool_result(compact_result)
                step.details = {
                    **step.details,
                    **self._build_task_terminal_details(task),
                }
                run.final_state = "running"
                run.message = step.message or run.message
                run.checkpoint = {
                    **run.checkpoint,
                    "phase": "decision",
                    "max_steps": max_steps,
                    "executed_signatures": self._dump_executed_signatures(executed_signatures),
                }
                run.checkpoint.pop("current_tool", None)
                self._store.save_run(run)
                return
            if task["state"] == "cancelled":
                step.status = "cancelled"
                step.message = compact_result.get("error") or f"task cancelled: {task_id}"
                step.details = {
                    **step.details,
                    **self._build_task_terminal_details(task),
                }
                self._store.save_run(run)
                self._finalize_run(run, state="cancelled", message=step.message)
                return
            step.status = "failed"
            step.message = compact_result.get("error") or f"task failed: {task_id}"
            step.details = {
                **step.details,
                **self._build_task_terminal_details(task),
            }
            self._store.save_run(run)
            self._finalize_run(run, state="failed", message=step.message)
            return

    def _restore_executed_signatures(
        self,
        checkpoint: dict,
    ) -> set[tuple[str, tuple[tuple[str, str], ...]]]:
        restored: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        payload = checkpoint.get("executed_signatures") if isinstance(checkpoint, dict) else None
        if not isinstance(payload, list):
            return restored
        for item in payload:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool_name")
            arguments = item.get("arguments")
            if not isinstance(tool_name, str) or not isinstance(arguments, list):
                continue
            restored.add(
                (
                    tool_name,
                    tuple(
                        (str(pair.get("key") or ""), str(pair.get("value") or ""))
                        for pair in arguments
                        if isinstance(pair, dict)
                    ),
                )
            )
        return restored

    def _dump_executed_signatures(
        self,
        signatures: set[tuple[str, tuple[tuple[str, str], ...]]],
    ) -> list[dict]:
        dumped: list[dict] = []
        for tool_name, arguments in sorted(signatures):
            dumped.append(
                {
                    "tool_name": tool_name,
                    "arguments": [
                        {"key": str(key), "value": str(value)}
                        for key, value in arguments
                    ],
                }
            )
        return dumped

    def _resume_interrupted_tool_step(
        self,
        run: AgentRunRecord,
        *,
        max_steps: int | None = None,
        executed_signatures: set[tuple[str, tuple[tuple[str, str], ...]]] | None = None,
    ) -> bool:
        current_tool = run.checkpoint.get("current_tool") if isinstance(run.checkpoint, dict) else None
        if not isinstance(current_tool, dict):
            return False
        tool_name = current_tool.get("tool_name")
        tool_arguments = current_tool.get("tool_arguments")
        task_id = current_tool.get("task_id")
        step_id = current_tool.get("step_id")
        if not isinstance(tool_name, str) or not isinstance(tool_arguments, dict):
            return False
        step = next((item for item in run.steps if item.step_id == step_id), None)
        if step is None:
            step = self._start_step(run, kind="tool", title=f"Resume {tool_name}", tool_name=tool_name)
        step.status = "running"
        step.finished_at = None
        step.details = {
            **step.details,
            "request": dict(tool_arguments),
            "resumed": True,
        }
        run.final_state = "waiting_task" if task_id else "running"
        run.updated_at = _now_iso()
        self._store.save_run(run)
        if isinstance(task_id, str) and task_id:
            self._wait_for_async_task(
                run,
                step,
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                task_id=task_id,
                max_steps=max_steps or get_settings().agent_max_steps,
                executed_signatures=executed_signatures or set(),
            )
            return True
        self._execute_tool_step(
            run,
            tool_name,
            tool_arguments,
            max_steps=max_steps or get_settings().agent_max_steps,
            executed_signatures=executed_signatures or set(),
            resume=True,
        )
        return True

    def _start_step(
        self,
        run: AgentRunRecord,
        *,
        kind: str,
        title: str,
        tool_name: str | None = None,
    ) -> AgentStepRecord:
        step = AgentStepRecord(
            step_id=self._store.create_step_id(),
            step_index=len(run.steps) + 1,
            kind=kind,
            status="running",
            title=title,
            tool_name=tool_name,
            started_at=_now_iso(),
        )
        run.steps.append(step)
        run.updated_at = _now_iso()
        self._store.save_run(run)
        return step

    def _record_plan_step(
        self,
        run: AgentRunRecord,
        payload: AgentChatRequest,
        prior_runs: list[AgentRunRecord],
        *,
        resume: bool = False,
    ) -> None:
        if resume and run.steps and run.steps[0].kind == "plan":
            step = run.steps[0]
            step.status = "completed"
        else:
            step = self._start_step(run, kind="plan", title="Frame objective and working plan")
        step.status = "completed"
        step.message = run.plan_summary or self._build_initial_plan_summary(payload, prior_runs)
        step.details = {
            "trigger_kind": run.trigger_kind,
            "parent_run_id": run.parent_run_id,
            "root_run_id": run.root_run_id,
            "max_steps": payload.max_steps or get_settings().agent_max_steps,
            "resumed": resume,
        }
        step.finished_at = _now_iso()
        run.updated_at = _now_iso()
        run.checkpoint = {
            **run.checkpoint,
            "phase": "decision" if not payload.tool_name else "tool",
            "resume_required": False,
        }
        self._store.save_run(run)

    def _execute_langgraph_pipeline(self, run: AgentRunRecord, payload: AgentChatRequest) -> None:
        run.provider = "langgraph"
        run.model = "deterministic-preprocess-pipeline"
        run.checkpoint = {
            **run.checkpoint,
            "engine": "langgraph",
            "phase": "graph_running",
        }
        self._store.save_run(run)
        pipeline_state = run_pipeline(payload.message)
        tool_calls = pipeline_state.get("tool_calls", [])
        for tool_call in tool_calls:
            step = self._start_step(run, kind="tool", title=f"Execute {tool_call.name}", tool_name=tool_call.name)
            step.tool_call = tool_call
            step.status = "failed" if tool_call.error else "completed"
            step.message = tool_call.error or self._summarize_tool_result(tool_call.result)
            step.details = {
                "request": dict(tool_call.arguments),
                "langgraph": True,
            }
            step.finished_at = _now_iso()
            run.tool_calls.append(tool_call)
            run.message = step.message or run.message
            run.updated_at = _now_iso()
            self._store.save_run(run)
            if tool_call.error:
                self._finalize_run(run, state="failed", message=tool_call.error)
                return
        final_output_dir = pipeline_state.get("current_output_dir") or pipeline_state.get("current_dataset_dir")
        run.plan_summary = self._build_langgraph_plan_summary(payload.message)
        self._finalize_run(
            run,
            state="completed",
            message=f"LangGraph pipeline completed; output={final_output_dir}" if final_output_dir else "LangGraph pipeline completed.",
        )

    def _build_langgraph_plan_summary(self, message: str) -> str:
        lowered = message.lower()
        steps: list[str] = []
        if "xml转yolo" in lowered or "xml to yolo" in lowered or "xml-to-yolo" in lowered:
            steps.append("xml-to-yolo")
        if "全转为0" in lowered or "转为0" in lowered or "reset" in lowered:
            steps.append("reset-yolo-label-index")
        if "train_only" in lowered:
            steps.append("split-yolo-dataset(train_only)")
        if "滑窗" in lowered or "裁剪" in lowered or "crop" in lowered:
            steps.append("yolo-sliding-window-crop")
        if "增强" in lowered or "augment" in lowered:
            steps.append("yolo-augment")
        return "LangGraph pipeline: " + " -> ".join(steps) if steps else "LangGraph pipeline"

    def _finalize_run(self, run: AgentRunRecord, *, state: str, message: str) -> None:
        run.final_state = state
        run.message = message
        run.updated_at = _now_iso()
        run.finished_at = _now_iso()
        run.checkpoint = {
            **run.checkpoint,
            "phase": "finished",
            "resume_required": False,
        }
        run.checkpoint.pop("current_tool", None)
        self._store.save_run(run)

    def _ensure_run_active(self, run_id: str) -> None:
        run = self._store.get_run(run_id)
        if run is None:
            raise AgentRunCancelledError(f"agent run not found: {run_id}")
        if run.cancellation_requested:
            last_task_id = None
            for step in reversed(run.steps):
                if step.task_id:
                    last_task_id = step.task_id
                    break
            if last_task_id:
                cancel_task(last_task_id)
            raise AgentRunCancelledError(f"agent run cancelled: {run_id}")

    def _format_task_progress(self, task: dict) -> str | None:
        progress_message = task.get("progress_message")
        current = task.get("progress_current")
        total = task.get("progress_total")
        if progress_message and current is not None and total is not None:
            return f"{progress_message} ({current}/{total})"
        if progress_message:
            return str(progress_message)
        if current is not None and total is not None:
            return f"task progress: {current}/{total}"
        return None

    def _format_async_submission_message(self, tool_name: str) -> str:
        return f"Preparing to run {tool_name}."

    def _format_async_resubmission_message(self, tool_name: str) -> str:
        return f"Service restarted. Resubmitted {tool_name} and waiting for it to start."

    def _format_task_wait_message(
        self,
        tool_name: str,
        task: dict,
        *,
        progress_text: str | None,
    ) -> str:
        state = str(task.get("state") or "").lower()
        if progress_text:
            if state == "pending":
                return f"{tool_name} is queued. {progress_text}"
            return f"Running {tool_name}. {progress_text}"
        if state == "pending":
            return f"{tool_name} is queued and waiting for a worker."
        if state == "running":
            return f"Running {tool_name}."
        return f"Waiting on {tool_name}."

    def _build_task_wait_details(self, task: dict) -> dict:
        events = task.get("events") if isinstance(task.get("events"), list) else []
        return {
            "task_state": task.get("state"),
            "progress_current": task.get("progress_current"),
            "progress_total": task.get("progress_total"),
            "progress_message": task.get("progress_message"),
            "recent_events": events[-3:],
        }

    def _build_task_terminal_details(self, task: dict) -> dict:
        details = self._build_task_wait_details(task)
        details.update(
            {
                "finished_at": task.get("finished_at"),
                "error": task.get("error"),
            }
        )
        return details

    def _compact_task_result(self, task: dict) -> dict:
        result = task.get("result")
        compact_result = dict(result) if isinstance(result, dict) else result
        if isinstance(compact_result, dict) and isinstance(compact_result.get("details"), list):
            compact_result["details_count"] = len(compact_result["details"])
            compact_result.pop("details", None)
        return {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "state": task["state"],
            "result": compact_result,
            "error": task.get("error"),
            "progress_current": task.get("progress_current"),
            "progress_total": task.get("progress_total"),
            "progress_message": task.get("progress_message"),
        }

    def _resolve_parent_run(
        self,
        payload: AgentChatRequest,
        prior_runs: list[AgentRunRecord],
    ) -> AgentRunRecord | None:
        if payload.parent_run_id:
            for run in reversed(prior_runs):
                if run.run_id == payload.parent_run_id:
                    return run
            return None
        return None

    def _build_initial_plan_summary(
        self,
        payload: AgentChatRequest,
        prior_runs: list[AgentRunRecord],
    ) -> str:
        resources = self._build_session_resource_context(prior_runs)
        context_bits: list[str] = []
        if resources.dataset_root:
            context_bits.append(f"dataset_root={resources.dataset_root}")
        if resources.labels_dir:
            context_bits.append(f"labels_dir={resources.labels_dir}")
        if resources.images_dir:
            context_bits.append(f"images_dir={resources.images_dir}")
        base = f"Goal: {payload.message.strip()}"
        if context_bits:
            base += f" | Reusable context: {', '.join(context_bits)}"
        return base

    def _decide_action(
        self,
        *,
        payload: AgentChatRequest,
        provider,
        settings,
        prior_runs: list[AgentRunRecord],
        current_run: AgentRunRecord | None,
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

        max_attempts = 3
        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return (
                    request_tool_decision(
                        settings=settings,
                        provider=provider,
                        message=payload.message,
                        tools=get_executable_tool_specs(),
                        conversation_context=self._build_conversation_context(prior_runs, current_run=current_run),
                    ),
                    None,
                )
            except ProviderCallError as exc:
                last_error = str(exc)
                if attempt >= max_attempts or not self._is_retryable_provider_error(last_error):
                    return None, last_error
                self._record_provider_retry(
                    current_run,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error_message=last_error,
                )
                time.sleep(0.2 * attempt)
        return None, last_error or "provider decision failed"

    def _is_retryable_provider_error(self, error_message: str) -> bool:
        retryable_signals = (
            "empty message content",
            "returned 429",
            "timed out",
            "failed to reach provider",
            "returned no choices",
            "invalid json payload",
            "non-object json payload",
        )
        lowered = error_message.lower()
        return any(signal in lowered for signal in retryable_signals)

    def _record_provider_retry(
        self,
        run: AgentRunRecord | None,
        *,
        attempt: int,
        max_attempts: int,
        error_message: str,
    ) -> None:
        if run is None:
            return
        run.message = (
            f"provider transient failure during decision; retrying "
            f"({attempt}/{max_attempts - 1}): {error_message}"
        )
        run.updated_at = _now_iso()
        run.checkpoint = {
            **run.checkpoint,
            "phase": "decision_retry",
            "provider_retry_attempt": attempt,
            "provider_retry_error": error_message,
        }
        self._store.save_run(run)

    def _normalize_tool_arguments(self, definition, arguments: dict) -> dict:
        normalized = dict(arguments)
        if definition.normalize_arguments is not None:
            normalized = definition.normalize_arguments(normalized)
        return normalized

    def _prepare_tool_execution_from_decision(
        self,
        user_message: str,
        decision: LLMToolDecision,
        tool_calls: list[ToolCallRecord],
        executed_signatures: set[tuple[str, tuple[tuple[str, str], ...]]],
        prior_runs: list[AgentRunRecord],
    ) -> tuple[str, dict, tuple[str, tuple[tuple[str, str], ...]]] | None:
        tool_name = decision.tool_name
        if not tool_name:
            return None
        tool_arguments = self._hydrate_tool_arguments(
            tool_name,
            dict(decision.tool_arguments),
            prior_runs,
        )
        definition = get_tool_definition(tool_name)
        if definition is None:
            return None
        normalized_arguments = self._normalize_tool_arguments(definition, tool_arguments)
        signature = (
            tool_name,
            tuple(sorted((str(key), str(value)) for key, value in normalized_arguments.items())),
        )
        if signature not in executed_signatures:
            return tool_name, normalized_arguments, signature

        fallback = self._infer_followup_decision_from_history(
            user_message,
            tool_calls,
            executed_signatures,
        )
        if fallback is None or not fallback.tool_name:
            return None
        fallback_definition = get_tool_definition(fallback.tool_name)
        if fallback_definition is None:
            return None
        fallback_arguments = self._normalize_tool_arguments(
            fallback_definition,
            dict(fallback.tool_arguments),
        )
        fallback_signature = (
            fallback.tool_name,
            tuple(sorted((str(key), str(value)) for key, value in fallback_arguments.items())),
        )
        if fallback_signature in executed_signatures:
            return None
        return fallback.tool_name, fallback_arguments, fallback_signature

    def _infer_followup_decision_from_history(
        self,
        user_message: str,
        tool_calls: list[ToolCallRecord],
        executed_signatures: set[tuple[str, tuple[tuple[str, str], ...]]],
    ) -> LLMToolDecision | None:
        if not tool_calls:
            return None
        message = user_message.lower()
        last_tool_call = tool_calls[-1]
        output_dir = self._extract_output_dir_from_tool_call(last_tool_call)

        if (
            last_tool_call.name == "yolo-sliding-window-crop"
            and output_dir
            and any(keyword in message for keyword in ("增强", "augment"))
        ):
            fallback = LLMToolDecision(
                action="execute",
                tool_name="yolo-augment",
                tool_arguments={"input_dir": output_dir},
            )
            return self._materialize_fallback_decision(fallback, executed_signatures)

        if (
            last_tool_call.name == "split-yolo-dataset"
            and output_dir
            and any(keyword in message for keyword in ("滑窗", "裁剪", "crop"))
        ):
            fallback = LLMToolDecision(
                action="execute",
                tool_name="yolo-sliding-window-crop",
                tool_arguments={"input_dir": output_dir},
            )
            return self._materialize_fallback_decision(fallback, executed_signatures)

        return None

    def _materialize_fallback_decision(
        self,
        decision: LLMToolDecision,
        executed_signatures: set[tuple[str, tuple[tuple[str, str], ...]]],
    ) -> LLMToolDecision | None:
        tool_name = decision.tool_name
        if not tool_name:
            return None
        definition = get_tool_definition(tool_name)
        if definition is None:
            return None
        normalized_arguments = self._normalize_tool_arguments(definition, dict(decision.tool_arguments))
        signature = (
            tool_name,
            tuple(sorted((str(key), str(value)) for key, value in normalized_arguments.items())),
        )
        if signature in executed_signatures:
            return None
        return LLMToolDecision(
            action="execute",
            tool_name=tool_name,
            tool_arguments=normalized_arguments,
            message=decision.message,
            plan_summary=decision.plan_summary,
        )

    def _extract_output_dir_from_tool_call(self, tool_call: ToolCallRecord) -> str | None:
        result = tool_call.result if isinstance(tool_call.result, dict) else {}
        nested_result = result.get("result") if isinstance(result.get("result"), dict) else {}
        for source in (nested_result, result, tool_call.arguments):
            value = source.get("output_dir") if isinstance(source, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

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
        raw_output_dir = self._first_path_value(
            task_result,
            result,
            keys=("output_dir",),
        )
        raw_input_dir = self._first_path_value(
            task_result,
            result,
            tool_call.arguments,
            keys=("input_dir", "dataset_root", "source_dataset_root"),
        )
        dataset_root = raw_output_dir or raw_input_dir
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
        if raw_output_dir and raw_output_dir.rstrip("/").endswith("/labels"):
            labels_dir = labels_dir or raw_output_dir
            dataset_root = self._parent_dataset_root(raw_output_dir)
        elif raw_output_dir and raw_output_dir.rstrip("/").endswith("/images"):
            images_dir = images_dir or raw_output_dir
            dataset_root = self._parent_dataset_root(raw_output_dir)
        elif raw_input_dir and raw_input_dir.rstrip("/").endswith("/labels"):
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

    def _build_conversation_context(
        self,
        prior_runs: list[AgentRunRecord],
        *,
        current_run: AgentRunRecord | None,
    ) -> str | None:
        resources = self._build_session_resource_context(
            prior_runs + ([current_run] if current_run is not None else [])
        )
        lines: list[str] = []
        if resources.dataset_root or resources.labels_dir or resources.images_dir:
            lines.append("Recent session resources:")
            if resources.dataset_root:
                lines.append(f"- dataset_root: {resources.dataset_root}")
            if resources.labels_dir:
                lines.append(f"- labels_dir: {resources.labels_dir}")
            if resources.images_dir:
                lines.append(f"- images_dir: {resources.images_dir}")
            lines.append(
                "If the current request omits a path, choose the resource type that best fits the selected tool."
            )
        if current_run is not None and current_run.plan_summary:
            lines.append(f"Current plan summary: {current_run.plan_summary}")
        if current_run is not None and current_run.steps:
            lines.append("Current run progress:")
            for step in current_run.steps[-6:]:
                detail = f"- step {step.step_index} [{step.kind}/{step.status}] {step.title}"
                if step.tool_name:
                    detail += f" tool={step.tool_name}"
                if step.message:
                    detail += f" message={step.message}"
                lines.append(detail)
        return "\n".join(lines) if lines else None

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
