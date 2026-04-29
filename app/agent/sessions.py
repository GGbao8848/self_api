import json
import sqlite3
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.agent.types import AgentRunRecord, ToolCallRecord
from app.core.config import get_settings


class SQLiteAgentSessionStore:
    def __init__(self) -> None:
        self._lock = Lock()

    def create_session_id(self) -> str:
        return uuid4().hex

    def create_run_id(self) -> str:
        return uuid4().hex

    def save_run(self, run: AgentRunRecord) -> None:
        with self._lock:
            self._ensure_schema()
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO agent_runs (
                        run_id,
                        session_id,
                        user_message,
                        message,
                        final_state,
                        provider,
                        model,
                        tool_calls_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.run_id,
                        run.session_id,
                        run.user_message,
                        run.message,
                        run.final_state,
                        run.provider,
                        run.model,
                        json.dumps(
                            [
                                {
                                    "name": tool_call.name,
                                    "arguments": tool_call.arguments,
                                    "result": tool_call.result,
                                    "error": tool_call.error,
                                }
                                for tool_call in run.tool_calls
                            ],
                            ensure_ascii=True,
                        ),
                    ),
                )

    def get_run(self, run_id: str) -> AgentRunRecord | None:
        with self._lock:
            self._ensure_schema()
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT
                        run_id,
                        session_id,
                        user_message,
                        message,
                        final_state,
                        provider,
                        model,
                        tool_calls_json
                    FROM agent_runs
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def list_session_runs(self, session_id: str) -> list[AgentRunRecord]:
        with self._lock:
            self._ensure_schema()
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        run_id,
                        session_id,
                        user_message,
                        message,
                        final_state,
                        provider,
                        model,
                        tool_calls_json
                    FROM agent_runs
                    WHERE session_id = ?
                    ORDER BY created_index ASC
                    """,
                    (session_id,),
                ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def list_sessions(self) -> list[tuple[str, list[AgentRunRecord]]]:
        with self._lock:
            self._ensure_schema()
            with self._connect() as connection:
                session_rows = connection.execute(
                    """
                    SELECT session_id
                    FROM agent_runs
                    GROUP BY session_id
                    ORDER BY MAX(created_index) DESC
                    """
                ).fetchall()
                session_ids = [str(row["session_id"]) for row in session_rows]
                runs_by_session = {
                    session_id: [
                        self._row_to_run(run_row)
                        for run_row in connection.execute(
                            """
                            SELECT
                                run_id,
                                session_id,
                                user_message,
                                message,
                                final_state,
                                provider,
                                model,
                                tool_calls_json
                            FROM agent_runs
                            WHERE session_id = ?
                            ORDER BY created_index ASC
                            """,
                            (session_id,),
                        ).fetchall()
                    ]
                    for session_id in session_ids
                }
        return [(session_id, runs_by_session[session_id]) for session_id in session_ids]

    def clear(self) -> None:
        with self._lock:
            db_path = self._db_path()
            if db_path.exists():
                db_path.unlink()

    def _db_path(self) -> Path:
        settings = get_settings()
        root = settings.resolved_storage_root
        root.mkdir(parents=True, exist_ok=True)
        return root / "agent_sessions.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path())
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    created_index INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    message TEXT NOT NULL,
                    final_state TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    tool_calls_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_runs_session_created
                ON agent_runs (session_id, created_index)
                """
            )

    def _row_to_run(self, row: sqlite3.Row) -> AgentRunRecord:
        tool_calls_payload = json.loads(row["tool_calls_json"] or "[]")
        return AgentRunRecord(
            session_id=str(row["session_id"]),
            run_id=str(row["run_id"]),
            user_message=str(row["user_message"]),
            message=str(row["message"]),
            final_state=str(row["final_state"]),
            provider=row["provider"],
            model=row["model"],
            tool_calls=[
                ToolCallRecord(
                    name=str(item.get("name") or ""),
                    arguments=item.get("arguments") or {},
                    result=item.get("result"),
                    error=item.get("error"),
                )
                for item in tool_calls_payload
                if isinstance(item, dict)
            ],
        )


agent_session_store = SQLiteAgentSessionStore()
