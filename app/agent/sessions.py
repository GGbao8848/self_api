from threading import Lock
from uuid import uuid4

from app.agent.types import AgentRunRecord


class InMemoryAgentSessionStore:
    def __init__(self) -> None:
        self._runs_by_session: dict[str, list[AgentRunRecord]] = {}
        self._runs_by_id: dict[str, AgentRunRecord] = {}
        self._lock = Lock()

    def create_session_id(self) -> str:
        return uuid4().hex

    def create_run_id(self) -> str:
        return uuid4().hex

    def save_run(self, run: AgentRunRecord) -> None:
        with self._lock:
            self._runs_by_session.setdefault(run.session_id, []).append(run)
            self._runs_by_id[run.run_id] = run

    def get_run(self, run_id: str) -> AgentRunRecord | None:
        with self._lock:
            return self._runs_by_id.get(run_id)

    def list_session_runs(self, session_id: str) -> list[AgentRunRecord] | None:
        with self._lock:
            runs = self._runs_by_session.get(session_id)
            return list(runs) if runs is not None else None

    def clear(self) -> None:
        with self._lock:
            self._runs_by_session.clear()
            self._runs_by_id.clear()


agent_session_store = InMemoryAgentSessionStore()
