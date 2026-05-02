import json
import sqlite3
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.agent.types import AgentRunRecord, AgentStepRecord, ToolCallRecord
from app.core.config import get_settings


class SQLiteAgentSessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._initialized = False

    def create_session_id(self) -> str:
        return uuid4().hex

    def create_run_id(self) -> str:
        return uuid4().hex

    def create_step_id(self) -> str:
        return uuid4().hex

    def save_run(self, run: AgentRunRecord) -> None:
        with self._lock:
            self._ensure_schema()
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT created_index, created_at FROM agent_runs WHERE run_id = ?",
                    (run.run_id,),
                ).fetchone()
                created_at = run.created_at or (existing["created_at"] if existing is not None else None)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO agent_runs (
                        created_index,
                        run_id,
                        session_id,
                        user_message,
                        message,
                        final_state,
                        parent_run_id,
                        root_run_id,
                        trigger_kind,
                        plan_summary,
                        provider,
                        model,
                        created_at,
                        updated_at,
                        finished_at,
                    cancellation_requested,
                    tool_calls_json,
                    steps_json,
                    request_payload_json,
                    checkpoint_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        existing["created_index"] if existing is not None else None,
                        run.run_id,
                        run.session_id,
                        run.user_message,
                        run.message,
                        run.final_state,
                        run.parent_run_id,
                        run.root_run_id,
                        run.trigger_kind,
                        run.plan_summary,
                        run.provider,
                        run.model,
                        created_at,
                        run.updated_at,
                        run.finished_at,
                        int(run.cancellation_requested),
                        json.dumps([self._tool_call_to_dict(item) for item in run.tool_calls], ensure_ascii=True),
                        json.dumps([self._step_to_dict(item) for item in run.steps], ensure_ascii=True),
                        json.dumps(run.request_payload, ensure_ascii=True),
                        json.dumps(run.checkpoint, ensure_ascii=True),
                    ),
                )

    def get_run(self, run_id: str) -> AgentRunRecord | None:
        with self._lock:
            self._ensure_schema()
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM agent_runs WHERE run_id = ?",
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
                    SELECT * FROM agent_runs
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
                            SELECT * FROM agent_runs
                            WHERE session_id = ?
                            ORDER BY created_index ASC
                            """,
                            (session_id,),
                        ).fetchall()
                    ]
                    for session_id in session_ids
                }
        return [(session_id, runs_by_session[session_id]) for session_id in session_ids]

    def cancel_run(self, run_id: str) -> AgentRunRecord | None:
        with self._lock:
            self._ensure_schema()
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM agent_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    return None
                run = self._row_to_run(row)
                if run.final_state in {"completed", "failed", "cancelled", "interrupted", "clarification_required", "requires_provider"}:
                    return run
                updated_at = _now_iso()
                connection.execute(
                    """
                    UPDATE agent_runs
                    SET cancellation_requested = 1, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (updated_at, run_id),
                )
        return self.get_run(run_id)

    def clear(self) -> None:
        with self._lock:
            self._initialized = False
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
        if self._initialized:
            return
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
                    parent_run_id TEXT,
                    root_run_id TEXT,
                    trigger_kind TEXT NOT NULL DEFAULT 'new',
                    plan_summary TEXT,
                    provider TEXT,
                    model TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    finished_at TEXT,
                    cancellation_requested INTEGER NOT NULL DEFAULT 0,
                    tool_calls_json TEXT NOT NULL DEFAULT '[]',
                    steps_json TEXT NOT NULL DEFAULT '[]',
                    request_payload_json TEXT NOT NULL DEFAULT '{}',
                    checkpoint_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(agent_runs)").fetchall()
            }
            for column_name, ddl in {
                "created_at": "ALTER TABLE agent_runs ADD COLUMN created_at TEXT",
                "updated_at": "ALTER TABLE agent_runs ADD COLUMN updated_at TEXT",
                "finished_at": "ALTER TABLE agent_runs ADD COLUMN finished_at TEXT",
                "cancellation_requested": "ALTER TABLE agent_runs ADD COLUMN cancellation_requested INTEGER NOT NULL DEFAULT 0",
                "steps_json": "ALTER TABLE agent_runs ADD COLUMN steps_json TEXT NOT NULL DEFAULT '[]'",
                "parent_run_id": "ALTER TABLE agent_runs ADD COLUMN parent_run_id TEXT",
                "root_run_id": "ALTER TABLE agent_runs ADD COLUMN root_run_id TEXT",
                "trigger_kind": "ALTER TABLE agent_runs ADD COLUMN trigger_kind TEXT NOT NULL DEFAULT 'new'",
                "plan_summary": "ALTER TABLE agent_runs ADD COLUMN plan_summary TEXT",
                "request_payload_json": "ALTER TABLE agent_runs ADD COLUMN request_payload_json TEXT NOT NULL DEFAULT '{}'",
                "checkpoint_json": "ALTER TABLE agent_runs ADD COLUMN checkpoint_json TEXT NOT NULL DEFAULT '{}'",
            }.items():
                if column_name not in columns:
                    connection.execute(ddl)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_runs_session_created
                ON agent_runs (session_id, created_index)
                """
            )
            interrupted_at = _now_iso()
            rows = connection.execute(
                """
                SELECT run_id, steps_json, checkpoint_json
                FROM agent_runs
                WHERE final_state IN ('accepted', 'running', 'waiting_task')
                """
            ).fetchall()
            for row in rows:
                steps_payload = json.loads(row["steps_json"] or "[]")
                checkpoint_payload = json.loads(row["checkpoint_json"] or "{}")
                updated_steps = []
                for item in steps_payload:
                    if not isinstance(item, dict):
                        continue
                    updated_item = dict(item)
                    if updated_item.get("status") == "running":
                        updated_item["status"] = "interrupted"
                        updated_item["finished_at"] = updated_item.get("finished_at") or interrupted_at
                        details = updated_item.get("details")
                        if not isinstance(details, dict):
                            details = {}
                        details["resume_required"] = True
                        updated_item["details"] = details
                    updated_steps.append(updated_item)
                if not isinstance(checkpoint_payload, dict):
                    checkpoint_payload = {}
                checkpoint_payload["resume_required"] = True
                checkpoint_payload["interrupted_at"] = interrupted_at
                connection.execute(
                    """
                    UPDATE agent_runs
                    SET
                        final_state = 'interrupted',
                        message = 'agent run interrupted by service restart; resume is available',
                        updated_at = ?,
                        steps_json = ?,
                        checkpoint_json = ?
                    WHERE run_id = ?
                    """,
                    (
                        interrupted_at,
                        json.dumps(updated_steps, ensure_ascii=True),
                        json.dumps(checkpoint_payload, ensure_ascii=True),
                        str(row["run_id"]),
                    ),
                )
        self._initialized = True

    def _row_to_run(self, row: sqlite3.Row) -> AgentRunRecord:
        tool_calls_payload = json.loads(row["tool_calls_json"] or "[]")
        steps_payload = json.loads(row["steps_json"] or "[]")
        request_payload = json.loads(row["request_payload_json"] or "{}")
        checkpoint = json.loads(row["checkpoint_json"] or "{}")
        return AgentRunRecord(
            session_id=str(row["session_id"]),
            run_id=str(row["run_id"]),
            user_message=str(row["user_message"]),
            message=str(row["message"]),
            final_state=str(row["final_state"]),
            parent_run_id=row["parent_run_id"],
            root_run_id=row["root_run_id"],
            trigger_kind=str(row["trigger_kind"] or "new"),
            plan_summary=row["plan_summary"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
            cancellation_requested=bool(row["cancellation_requested"]),
            provider=row["provider"],
            model=row["model"],
            tool_calls=[self._dict_to_tool_call(item) for item in tool_calls_payload if isinstance(item, dict)],
            steps=[self._dict_to_step(item) for item in steps_payload if isinstance(item, dict)],
            request_payload=request_payload if isinstance(request_payload, dict) else {},
            checkpoint=checkpoint if isinstance(checkpoint, dict) else {},
        )

    def _tool_call_to_dict(self, tool_call: ToolCallRecord) -> dict:
        return {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "result": tool_call.result,
            "error": tool_call.error,
        }

    def _dict_to_tool_call(self, item: dict) -> ToolCallRecord:
        return ToolCallRecord(
            name=str(item.get("name") or ""),
            arguments=item.get("arguments") or {},
            result=item.get("result"),
            error=item.get("error"),
        )

    def _step_to_dict(self, step: AgentStepRecord) -> dict:
        return {
            "step_id": step.step_id,
            "step_index": step.step_index,
            "kind": step.kind,
            "status": step.status,
            "title": step.title,
            "message": step.message,
            "details": step.details,
            "tool_name": step.tool_name,
            "task_id": step.task_id,
            "task_type": step.task_type,
            "started_at": step.started_at,
            "finished_at": step.finished_at,
            "tool_call": self._tool_call_to_dict(step.tool_call) if step.tool_call is not None else None,
        }

    def _dict_to_step(self, item: dict) -> AgentStepRecord:
        tool_call_payload = item.get("tool_call")
        return AgentStepRecord(
            step_id=str(item.get("step_id") or ""),
            step_index=int(item.get("step_index") or 0),
            kind=str(item.get("kind") or ""),
            status=str(item.get("status") or ""),
            title=str(item.get("title") or ""),
            message=item.get("message"),
            details=item.get("details") if isinstance(item.get("details"), dict) else {},
            tool_name=item.get("tool_name"),
            task_id=item.get("task_id"),
            task_type=item.get("task_type"),
            started_at=item.get("started_at"),
            finished_at=item.get("finished_at"),
            tool_call=self._dict_to_tool_call(tool_call_payload) if isinstance(tool_call_payload, dict) else None,
        )


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


agent_session_store = SQLiteAgentSessionStore()
