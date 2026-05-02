from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.langgraph_pipeline import matches_langgraph_pipeline_request
from app.agent.types import AgentRunRecord, LLMToolDecision, ProviderSelection
from app.schemas.agent import AgentChatRequest

if TYPE_CHECKING:
    from app.agent.runtime import AgentRuntime


class AgentGraphState(TypedDict, total=False):
    run_mode: str
    phase: str
    route: str
    session_id: str
    run_id: str
    user_message: str
    provider_name: str | None
    provider_configured: bool
    resume_requested: bool
    explicit_tool_requested: bool
    deterministic_pipeline_requested: bool
    step_count: int
    max_steps: int
    decision: LLMToolDecision | None
    current_tool_name: str | None
    current_tool_arguments: dict
    terminal_state: str | None
    terminal_message: str | None
    executed_signatures: set[tuple[str, tuple[tuple[str, str], ...]]]


class LangGraphAgentExecutor:
    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        run: AgentRunRecord,
        payload: AgentChatRequest,
        prior_runs: list[AgentRunRecord],
        provider: ProviderSelection,
        resume: bool = False,
    ) -> None:
        self._runtime = runtime
        self._run = run
        self._payload = payload
        self._prior_runs = prior_runs
        self._provider = provider
        self._resume = resume

    def execute(self) -> None:
        graph = self._build_graph().compile()
        graph.invoke(
            AgentGraphState(
                run_mode="inline" if not self._payload.async_run else "long_run",
                phase="graph_invoked",
                route="initialize",
                session_id=self._run.session_id,
                run_id=self._run.run_id,
                user_message=self._payload.message,
                provider_name=self._provider.provider or None,
                provider_configured=self._provider.configured,
                resume_requested=self._resume,
                explicit_tool_requested=bool(self._payload.tool_name),
                deterministic_pipeline_requested=matches_langgraph_pipeline_request(self._payload.message),
                step_count=0,
                max_steps=self._payload.max_steps or self._runtime._settings().agent_max_steps,
                decision=None,
                current_tool_name=None,
                current_tool_arguments={},
                terminal_state=None,
                terminal_message=None,
                executed_signatures=self._runtime._restore_executed_signatures(self._run.checkpoint),
            )
        )

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentGraphState)
        graph.add_node("initialize", self._initialize_node)
        graph.add_node("resume_current_tool", self._resume_current_tool_node)
        graph.add_node("execute_explicit_tool", self._execute_explicit_tool_node)
        graph.add_node("execute_deterministic_pipeline", self._execute_deterministic_pipeline_node)
        graph.add_node("provider_missing", self._provider_missing_node)
        graph.add_node("decide_next_action", self._decide_next_action_node)
        graph.add_node("respond", self._respond_node)
        graph.add_node("clarify", self._clarify_node)
        graph.add_node("unsupported_action", self._unsupported_action_node)
        graph.add_node("execute_decision_tool", self._execute_decision_tool_node)

        graph.add_edge(START, "initialize")
        graph.add_conditional_edges(
            "initialize",
            self._route_after_initialize,
            {
                "resume_current_tool": "resume_current_tool",
                "execute_explicit_tool": "execute_explicit_tool",
                "execute_deterministic_pipeline": "execute_deterministic_pipeline",
                "provider_missing": "provider_missing",
                "decide_next_action": "decide_next_action",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "resume_current_tool",
            self._route_after_resume,
            {
                "decide_next_action": "decide_next_action",
                "end": END,
            },
        )
        graph.add_edge("execute_explicit_tool", END)
        graph.add_edge("execute_deterministic_pipeline", END)
        graph.add_edge("provider_missing", END)
        graph.add_conditional_edges(
            "decide_next_action",
            self._route_after_decision,
            {
                "respond": "respond",
                "clarify": "clarify",
                "execute_decision_tool": "execute_decision_tool",
                "unsupported_action": "unsupported_action",
                "end": END,
            },
        )
        graph.add_edge("respond", END)
        graph.add_edge("clarify", END)
        graph.add_edge("unsupported_action", END)
        graph.add_conditional_edges(
            "execute_decision_tool",
            self._route_after_tool_execution,
            {
                "decide_next_action": "decide_next_action",
                "end": END,
            },
        )
        return graph

    def _initialize_node(self, state: AgentGraphState) -> AgentGraphState:
        state["phase"] = "graph_initializing"
        self._persist_graph_state(state)
        return state

    def _route_after_initialize(self, state: AgentGraphState) -> str:
        current_tool = self._run.checkpoint.get("current_tool") if isinstance(self._run.checkpoint, dict) else None
        if self._resume and isinstance(current_tool, dict):
            state["route"] = "resume_current_tool"
            return "resume_current_tool"
        if self._payload.tool_name:
            state["route"] = "execute_explicit_tool"
            return "execute_explicit_tool"
        if matches_langgraph_pipeline_request(self._payload.message):
            state["route"] = "execute_deterministic_pipeline"
            return "execute_deterministic_pipeline"
        if not self._provider.configured:
            state["route"] = "provider_missing"
            return "provider_missing"
        state["route"] = "decide_next_action"
        return "decide_next_action"

    def _resume_current_tool_node(self, state: AgentGraphState) -> AgentGraphState:
        state["phase"] = "graph_resuming_tool"
        self._runtime._resume_interrupted_tool_step(
            self._run,
            max_steps=state["max_steps"],
            executed_signatures=state["executed_signatures"],
        )
        return state

    def _route_after_resume(self, state: AgentGraphState) -> str:
        if self._run.final_state in {"failed", "cancelled"}:
            return "end"
        if self._payload.tool_name:
            last_call = self._run.tool_calls[-1] if self._run.tool_calls else None
            self._runtime._finalize_run(
                self._run,
                state="completed",
                message=self._runtime._summarize_tool_result(last_call.result if last_call else None),
            )
            return "end"
        return "decide_next_action"

    def _execute_explicit_tool_node(self, state: AgentGraphState) -> AgentGraphState:
        state["phase"] = "graph_explicit_tool"
        self._persist_graph_state(state)
        self._runtime._execute_explicit_tool_step(
            self._run,
            self._payload,
            self._prior_runs,
            resume=self._resume,
        )
        if self._run.final_state in {"completed", "failed", "cancelled"}:
            state["terminal_state"] = self._run.final_state
            state["terminal_message"] = self._run.message
            state["phase"] = "graph_explicit_tool_finished"
            self._persist_graph_state(state)
        return state

    def _execute_deterministic_pipeline_node(self, state: AgentGraphState) -> AgentGraphState:
        state["phase"] = "graph_deterministic_pipeline"
        self._persist_graph_state(state)
        self._runtime._execute_langgraph_pipeline(self._run, self._payload)
        if self._run.final_state in {"completed", "failed", "cancelled"}:
            state["terminal_state"] = self._run.final_state
            state["terminal_message"] = self._run.message
            state["phase"] = "graph_deterministic_pipeline_finished"
            self._persist_graph_state(state)
        return state

    def _provider_missing_node(self, state: AgentGraphState) -> AgentGraphState:
        state["phase"] = "graph_provider_missing"
        state["terminal_state"] = "requires_provider"
        state["terminal_message"] = self._provider.reason or "LLM provider is not configured"
        self._persist_graph_state(state)
        self._runtime._finalize_run(
            self._run,
            state=state["terminal_state"],
            message=state["terminal_message"],
        )
        return state

    def _decide_next_action_node(self, state: AgentGraphState) -> AgentGraphState:
        state["phase"] = "graph_decision"
        if state["step_count"] >= state["max_steps"]:
            state["terminal_state"] = "failed"
            state["terminal_message"] = (
                f"agent exceeded max_steps={state['max_steps']} before reaching a final answer"
            )
            self._runtime._finalize_run(
                self._run,
                state=state["terminal_state"],
                message=state["terminal_message"],
            )
            return state
        self._persist_graph_state(state)
        decision = self._runtime._record_decision_step(
            self._run,
            self._payload,
            self._provider,
            self._runtime._settings(),
            self._prior_runs,
        )
        state["step_count"] += 1
        state["decision"] = decision
        return state

    def _route_after_decision(self, state: AgentGraphState) -> str:
        if self._run.final_state in {"failed", "cancelled"}:
            return "end"
        decision = state.get("decision")
        if decision is None:
            return "end"
        if decision.action == "respond":
            return "respond"
        if decision.action == "clarify":
            return "clarify"
        if decision.action == "execute" and decision.tool_name:
            return "execute_decision_tool"
        return "unsupported_action"

    def _respond_node(self, state: AgentGraphState) -> AgentGraphState:
        decision = state.get("decision")
        state["phase"] = "graph_respond"
        state["terminal_state"] = "completed"
        state["terminal_message"] = (decision.message if decision is not None else None) or "Task completed."
        self._persist_graph_state(state)
        self._runtime._finalize_run(
            self._run,
            state=state["terminal_state"],
            message=state["terminal_message"],
        )
        return state

    def _clarify_node(self, state: AgentGraphState) -> AgentGraphState:
        decision = state.get("decision")
        state["phase"] = "graph_clarify"
        state["terminal_state"] = "clarification_required"
        state["terminal_message"] = (decision.message if decision is not None else None) or "I need more details to continue."
        self._persist_graph_state(state)
        self._runtime._finalize_run(
            self._run,
            state=state["terminal_state"],
            message=state["terminal_message"],
        )
        return state

    def _unsupported_action_node(self, state: AgentGraphState) -> AgentGraphState:
        state["phase"] = "graph_unsupported_action"
        state["terminal_state"] = "failed"
        state["terminal_message"] = "model returned an unsupported action for long-running execution"
        self._persist_graph_state(state)
        self._runtime._finalize_run(
            self._run,
            state=state["terminal_state"],
            message=state["terminal_message"],
        )
        return state

    def _execute_decision_tool_node(self, state: AgentGraphState) -> AgentGraphState:
        state["phase"] = "graph_execute_tool"
        decision = state.get("decision")
        if decision is None or not decision.tool_name:
            state["terminal_state"] = "failed"
            state["terminal_message"] = "model returned no executable tool decision"
            self._runtime._finalize_run(
                self._run,
                state=state["terminal_state"],
                message=state["terminal_message"],
            )
            return state
        prepared = self._runtime._prepare_tool_execution_from_decision(
            self._payload.message,
            decision,
            self._run.tool_calls,
            state["executed_signatures"],
            self._prior_runs + [self._run],
        )
        if prepared is None:
            state["terminal_state"] = "failed"
            state["terminal_message"] = f"repeated tool decision detected: {decision.tool_name}"
            self._runtime._finalize_run(
                self._run,
                state=state["terminal_state"],
                message=state["terminal_message"],
            )
            return state
        tool_name, tool_arguments, signature = prepared
        state["current_tool_name"] = tool_name
        state["current_tool_arguments"] = dict(tool_arguments)
        state["executed_signatures"].add(signature)
        self._persist_graph_state(state)
        self._runtime._execute_tool_step(
            self._run,
            tool_name,
            tool_arguments,
            max_steps=state["max_steps"],
            executed_signatures=state["executed_signatures"],
            resume=self._resume,
        )
        return state

    def _route_after_tool_execution(self, state: AgentGraphState) -> str:
        if self._run.final_state in {"failed", "cancelled"}:
            return "end"
        if state.get("run_mode") == "inline":
            last_call = self._run.tool_calls[-1] if self._run.tool_calls else None
            state["phase"] = "graph_respond"
            state["terminal_state"] = "completed"
            state["terminal_message"] = self._runtime._summarize_tool_result(last_call.result if last_call else None)
            self._persist_graph_state(state)
            self._runtime._finalize_run(
                self._run,
                state=state["terminal_state"],
                message=state["terminal_message"],
            )
            return "end"
        state["route"] = "decide_next_action"
        return "decide_next_action"

    def _checkpoint_graph_state(self, state: AgentGraphState) -> dict:
        return {
            "run_mode": state.get("run_mode"),
            "phase": state.get("phase"),
            "route": state.get("route"),
            "session_id": state.get("session_id"),
            "run_id": state.get("run_id"),
            "user_message": state.get("user_message"),
            "provider_name": state.get("provider_name"),
            "provider_configured": state.get("provider_configured"),
            "resume_requested": state.get("resume_requested"),
            "explicit_tool_requested": state.get("explicit_tool_requested"),
            "deterministic_pipeline_requested": state.get("deterministic_pipeline_requested"),
            "step_count": state.get("step_count"),
            "max_steps": state.get("max_steps"),
            "current_tool_name": state.get("current_tool_name"),
            "current_tool_arguments": state.get("current_tool_arguments", {}),
            "terminal_state": state.get("terminal_state"),
            "terminal_message": state.get("terminal_message"),
            "executed_signatures": self._runtime._dump_executed_signatures(state["executed_signatures"]),
        }

    def _persist_graph_state(self, state: AgentGraphState) -> None:
        self._run.checkpoint = {
            **self._run.checkpoint,
            "engine": "langgraph",
            "graph_state": self._checkpoint_graph_state(state),
        }
        self._run.updated_at = self._runtime._now()
        self._runtime._save_run(self._run)
