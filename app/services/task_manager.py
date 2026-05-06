import json
import logging
import sqlite3
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import get_settings
from app.services.artifact_store import register_path, summarize_record

TaskState = Literal["pending", "running", "interrupted", "succeeded", "failed", "cancelled"]
CallbackState = Literal["pending", "running", "succeeded", "failed"]
_IDEMPOTENT_CALLBACK_SUCCESS_CODES = {409}
_UNSET = object()
_TASK_LOCAL = threading.local()
_LOCK = threading.Lock()
_CALLBACK_URL_LOCKS: dict[str, threading.Lock] = {}
_CALLBACK_URL_LOCKS_GUARD = threading.Lock()
_ACTIVE_THREADS: dict[str, threading.Thread] = {}
_ACTIVE_THREADS_GUARD = threading.Lock()
_INITIALIZED = False
logger = logging.getLogger(__name__)


class TaskCancelledError(RuntimeError):
    pass


class TaskRecord(TypedDict):
    task_id: str
    task_type: str
    state: TaskState
    created_at: str
    updated_at: str
    finished_at: str | None
    result: dict[str, Any] | None
    error: str | None
    cancellation_requested: bool
    callback_url: str | None
    callback_state: CallbackState
    callback_sent_at: str | None
    callback_status_code: int | None
    callback_error: str | None
    callback_events: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    progress_current: int | None
    progress_total: int | None
    progress_message: str | None
    events: list[dict[str, Any]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    root = get_settings().resolved_storage_root
    root.mkdir(parents=True, exist_ok=True)
    return root / "tasks.sqlite3"


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_db_path(), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_initialized() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _LOCK:
        if _INITIALIZED:
            return
        with _connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    created_index INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL UNIQUE,
                    task_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    result_json TEXT,
                    error TEXT,
                    cancellation_requested INTEGER NOT NULL DEFAULT 0,
                    callback_url TEXT,
                    callback_state TEXT NOT NULL DEFAULT 'succeeded',
                    callback_sent_at TEXT,
                    callback_status_code INTEGER,
                    callback_error TEXT,
                    callback_events_json TEXT NOT NULL DEFAULT '[]',
                    artifacts_json TEXT NOT NULL DEFAULT '[]',
                    progress_current INTEGER,
                    progress_total INTEGER,
                    progress_message TEXT,
                    events_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_type_created
                ON tasks (task_type, created_index DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_state_created
                ON tasks (state, created_index DESC)
                """
            )
            interrupted_at = _now_iso()
            interrupted_error = "task interrupted by service restart before completion"
            connection.execute(
                """
                UPDATE tasks
                SET
                    state = 'interrupted',
                    error = COALESCE(NULLIF(error, ''), ?),
                    updated_at = ?,
                    events_json = json_insert(
                        COALESCE(events_json, '[]'),
                        '$[#]',
                        json_object(
                            'event_type', 'interrupted',
                            'message', ?,
                            'details', json('{}'),
                            'created_at', ?
                        )
                    )
                WHERE state IN ('pending', 'running')
                """,
                (interrupted_error, interrupted_at, interrupted_error, interrupted_at),
            )
        _INITIALIZED = True


def _set_current_task_id(task_id: str | None) -> None:
    _TASK_LOCAL.task_id = task_id


def get_current_task_id() -> str | None:
    return getattr(_TASK_LOCAL, "task_id", None)


def ensure_current_task_active() -> None:
    task_id = get_current_task_id()
    if not task_id:
        return
    task = get_task(task_id)
    if task is not None and task["cancellation_requested"]:
        raise TaskCancelledError(f"task cancelled: {task_id}")


def _row_to_task(row: sqlite3.Row) -> TaskRecord:
    result_payload = row["result_json"]
    callback_events_payload = row["callback_events_json"]
    artifacts_payload = row["artifacts_json"]
    events_payload = row["events_json"]
    return {
        "task_id": str(row["task_id"]),
        "task_type": str(row["task_type"]),
        "state": str(row["state"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "finished_at": row["finished_at"],
        "result": json.loads(result_payload) if result_payload else None,
        "error": row["error"],
        "cancellation_requested": bool(row["cancellation_requested"]),
        "callback_url": row["callback_url"],
        "callback_state": str(row["callback_state"]),
        "callback_sent_at": row["callback_sent_at"],
        "callback_status_code": row["callback_status_code"],
        "callback_error": row["callback_error"],
        "callback_events": json.loads(callback_events_payload or "[]"),
        "artifacts": json.loads(artifacts_payload or "[]"),
        "progress_current": row["progress_current"],
        "progress_total": row["progress_total"],
        "progress_message": row["progress_message"],
        "events": json.loads(events_payload or "[]"),
    }


def _get_task_locked(connection: sqlite3.Connection, task_id: str) -> TaskRecord | None:
    row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        return None
    return _row_to_task(row)


def _update_task(
    task_id: str,
    *,
    state: TaskState | None = None,
    finished_at: str | None | object = _UNSET,
    result: dict[str, Any] | None | object = _UNSET,
    error: str | None | object = _UNSET,
    cancellation_requested: bool | None = None,
    callback_state: CallbackState | None = None,
    callback_sent_at: str | None | object = _UNSET,
    callback_status_code: int | None | object = _UNSET,
    callback_error: str | None | object = _UNSET,
    callback_event: dict[str, Any] | object = _UNSET,
    artifacts: list[dict[str, Any]] | object = _UNSET,
    progress_current: int | None | object = _UNSET,
    progress_total: int | None | object = _UNSET,
    progress_message: str | None | object = _UNSET,
    event: dict[str, Any] | object = _UNSET,
) -> None:
    _ensure_initialized()
    with _LOCK:
        with _connect() as connection:
            task = _get_task_locked(connection, task_id)
            if task is None:
                return
            if state is not None:
                task["state"] = state
            if finished_at is not _UNSET:
                task["finished_at"] = finished_at
            if result is not _UNSET:
                task["result"] = result
            if error is not _UNSET:
                task["error"] = error
            if cancellation_requested is not None:
                task["cancellation_requested"] = cancellation_requested
            if callback_state is not None:
                task["callback_state"] = callback_state
            if callback_sent_at is not _UNSET:
                task["callback_sent_at"] = callback_sent_at
            if callback_status_code is not _UNSET:
                task["callback_status_code"] = callback_status_code
            if callback_error is not _UNSET:
                task["callback_error"] = callback_error
            if callback_event is not _UNSET:
                task["callback_events"].append(callback_event)
            if artifacts is not _UNSET:
                task["artifacts"] = artifacts
            if progress_current is not _UNSET:
                task["progress_current"] = progress_current
            if progress_total is not _UNSET:
                task["progress_total"] = progress_total
            if progress_message is not _UNSET:
                task["progress_message"] = progress_message
            if event is not _UNSET:
                task["events"].append(event)
            task["updated_at"] = _now_iso()
            connection.execute(
                """
                UPDATE tasks
                SET
                    state = ?,
                    updated_at = ?,
                    finished_at = ?,
                    result_json = ?,
                    error = ?,
                    cancellation_requested = ?,
                    callback_state = ?,
                    callback_sent_at = ?,
                    callback_status_code = ?,
                    callback_error = ?,
                    callback_events_json = ?,
                    artifacts_json = ?,
                    progress_current = ?,
                    progress_total = ?,
                    progress_message = ?,
                    events_json = ?
                WHERE task_id = ?
                """,
                (
                    task["state"],
                    task["updated_at"],
                    task["finished_at"],
                    json.dumps(task["result"], ensure_ascii=True) if task["result"] is not None else None,
                    task["error"],
                    int(task["cancellation_requested"]),
                    task["callback_state"],
                    task["callback_sent_at"],
                    task["callback_status_code"],
                    task["callback_error"],
                    json.dumps(task["callback_events"], ensure_ascii=True),
                    json.dumps(task["artifacts"], ensure_ascii=True),
                    task["progress_current"],
                    task["progress_total"],
                    task["progress_message"],
                    json.dumps(task["events"], ensure_ascii=True),
                    task_id,
                ),
            )


def append_task_event(
    task_id: str,
    *,
    event_type: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    _update_task(
        task_id,
        event={
            "event_type": event_type,
            "message": message,
            "details": details or {},
            "created_at": _now_iso(),
        },
    )


def update_task_progress(
    task_id: str,
    *,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> None:
    _update_task(
        task_id,
        progress_current=current,
        progress_total=total,
        progress_message=message,
        event={
            "event_type": "progress",
            "message": message or "progress updated",
            "details": {"current": current, "total": total},
            "created_at": _now_iso(),
        },
    )


def append_current_task_event(
    *,
    event_type: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    task_id = get_current_task_id()
    if not task_id:
        return
    append_task_event(
        task_id,
        event_type=event_type,
        message=message,
        details=details,
    )


def update_current_task_progress(
    *,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> None:
    task_id = get_current_task_id()
    if not task_id:
        return
    update_task_progress(
        task_id,
        current=current,
        total=total,
        message=message,
    )


def _post_callback(callback_url: str, payload: dict[str, Any], timeout: float) -> int:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        callback_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return int(response.getcode() or 0)


def _get_callback(callback_url: str, timeout: float) -> int:
    request = Request(callback_url, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return int(response.getcode() or 0)


def _build_callback_payload(task: TaskRecord) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "state": task["state"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "finished_at": task["finished_at"],
        "result": task["result"],
        "error": task["error"],
        "artifacts": task["artifacts"],
        "progress_current": task["progress_current"],
        "progress_total": task["progress_total"],
        "progress_message": task["progress_message"],
        "events": task["events"],
    }


def _get_callback_url_lock(callback_url: str) -> threading.Lock:
    with _CALLBACK_URL_LOCKS_GUARD:
        lock = _CALLBACK_URL_LOCKS.get(callback_url)
        if lock is None:
            lock = threading.Lock()
            _CALLBACK_URL_LOCKS[callback_url] = lock
        return lock


def _send_task_callback(task_id: str, timeout_seconds: float) -> None:
    task = get_task(task_id)
    if task is None or task["callback_url"] is None:
        return
    event_state = task["state"]
    event_time = _now_iso()
    _update_task(
        task_id,
        callback_state="running",
        callback_sent_at=None,
        callback_status_code=None,
        callback_error=None,
    )
    task = get_task(task_id)
    if task is None or task["callback_url"] is None:
        return

    callback_lock = _get_callback_url_lock(task["callback_url"])
    with callback_lock:
        method_used = "POST"
        try:
            try:
                status_code = _post_callback(
                    callback_url=task["callback_url"],
                    payload=_build_callback_payload(task),
                    timeout=timeout_seconds,
                )
            except HTTPError as exc:
                if exc.code in {405, 501}:
                    method_used = "GET"
                    status_code = _get_callback(
                        callback_url=task["callback_url"],
                        timeout=timeout_seconds,
                    )
                else:
                    raise

            callback_event = {
                "state": event_state,
                "attempted_at": event_time,
                "callback_url": task["callback_url"],
                "status_code": status_code,
                "method": method_used,
                "success": (200 <= status_code < 300)
                or (status_code in _IDEMPOTENT_CALLBACK_SUCCESS_CODES),
                "error": None if (200 <= status_code < 300) or (status_code in _IDEMPOTENT_CALLBACK_SUCCESS_CODES)
                else f"callback returned non-2xx status: {status_code}",
            }
            if callback_event["success"]:
                _update_task(
                    task_id,
                    callback_state="succeeded",
                    callback_sent_at=_now_iso(),
                    callback_status_code=status_code,
                    callback_error=None,
                    callback_event=callback_event,
                )
            else:
                _update_task(
                    task_id,
                    callback_state="failed",
                    callback_sent_at=_now_iso(),
                    callback_status_code=status_code,
                    callback_error=str(callback_event["error"]),
                    callback_event=callback_event,
                )
        except HTTPError as exc:
            success = exc.code in _IDEMPOTENT_CALLBACK_SUCCESS_CODES
            _update_task(
                task_id,
                callback_state="succeeded" if success else "failed",
                callback_sent_at=_now_iso(),
                callback_status_code=exc.code,
                callback_error=None if success else f"callback returned non-2xx status: {exc.code}",
                callback_event={
                    "state": event_state,
                    "attempted_at": event_time,
                    "callback_url": task["callback_url"],
                    "status_code": exc.code,
                    "method": method_used,
                    "success": success,
                    "error": None if success else f"callback returned non-2xx status: {exc.code}",
                },
            )
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            _update_task(
                task_id,
                callback_state="failed",
                callback_sent_at=_now_iso(),
                callback_status_code=None,
                callback_error=str(exc),
                callback_event={
                    "state": event_state,
                    "attempted_at": event_time,
                    "callback_url": task["callback_url"],
                    "status_code": None,
                    "method": method_used,
                    "success": False,
                    "error": str(exc),
                },
            )


def _extract_artifact_candidates(result: dict[str, Any]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for key, value in result.items():
        if not isinstance(value, str):
            continue
        if key.startswith("output_") or key in {
            "target_path",
            "labels_dir",
            "classes_file",
            "copied_classes_file",
        }:
            candidates.append((key, value))
    return candidates


def _register_task_artifacts(
    *,
    task_id: str,
    task_type: str,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for key, value in _extract_artifact_candidates(result):
        candidate = Path(value).expanduser().resolve(strict=False)
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            record = register_path(
                candidate,
                source=f"task:{task_type}:{key}",
                task_id=task_id,
                task_type=task_type,
            )
        except ValueError:
            continue
        artifacts.append(summarize_record(record))
    return artifacts


def submit_task(
    task_type: str,
    runner: Callable[[], dict[str, Any]],
    *,
    callback_url: str | None = None,
    callback_timeout_seconds: float = 10.0,
) -> str:
    _ensure_initialized()
    task_id = uuid.uuid4().hex
    now = _now_iso()
    callback_state: CallbackState = "pending" if callback_url else "succeeded"
    with _LOCK:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id,
                    task_type,
                    state,
                    created_at,
                    updated_at,
                    finished_at,
                    result_json,
                    error,
                    cancellation_requested,
                    callback_url,
                    callback_state,
                    callback_sent_at,
                    callback_status_code,
                    callback_error,
                    callback_events_json,
                    artifacts_json,
                    progress_current,
                    progress_total,
                    progress_message,
                    events_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    task_type,
                    "pending",
                    now,
                    now,
                    None,
                    None,
                    None,
                    0,
                    callback_url,
                    callback_state,
                    None,
                    None,
                    None,
                    "[]",
                    "[]",
                    None,
                    None,
                    None,
                    json.dumps(
                        [
                            {
                                "event_type": "submitted",
                                "message": f"task submitted: {task_type}",
                                "details": {},
                                "created_at": now,
                            }
                        ],
                        ensure_ascii=True,
                    ),
                ),
            )

    def _run() -> None:
        _set_current_task_id(task_id)
        append_task_event(task_id, event_type="started", message="task started")
        try:
            ensure_current_task_active()
            _update_task(task_id, state="running")
            task_result = runner()
            ensure_current_task_active()
        except TaskCancelledError as exc:
            _update_task(
                task_id,
                state="cancelled",
                error=str(exc),
                result=None,
                finished_at=_now_iso(),
                event={
                    "event_type": "cancelled",
                    "message": str(exc),
                    "details": {},
                    "created_at": _now_iso(),
                },
            )
            if callback_url:
                _send_task_callback(task_id, callback_timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            _update_task(
                task_id,
                state="failed",
                error=str(exc),
                result=None,
                finished_at=_now_iso(),
                event={
                    "event_type": "failed",
                    "message": str(exc),
                    "details": {},
                    "created_at": _now_iso(),
                },
            )
            if callback_url:
                _send_task_callback(task_id, callback_timeout_seconds)
        else:
            artifacts = _register_task_artifacts(
                task_id=task_id,
                task_type=task_type,
                result=task_result,
            )
            _update_task(
                task_id,
                state="succeeded",
                result=task_result,
                error=None,
                finished_at=_now_iso(),
                artifacts=artifacts,
                event={
                    "event_type": "completed",
                    "message": "task completed",
                    "details": {"artifacts_count": len(artifacts)},
                    "created_at": _now_iso(),
                },
            )
            if callback_url:
                _send_task_callback(task_id, callback_timeout_seconds)
        finally:
            _set_current_task_id(None)
            with _ACTIVE_THREADS_GUARD:
                _ACTIVE_THREADS.pop(task_id, None)

    thread = threading.Thread(
        target=_run,
        daemon=True,
        name=f"preprocess-task-{task_id[:8]}",
    )
    with _ACTIVE_THREADS_GUARD:
        _ACTIVE_THREADS[task_id] = thread
    thread.start()
    return task_id


def get_task(task_id: str) -> TaskRecord | None:
    _ensure_initialized()
    with _connect() as connection:
        task = _get_task_locked(connection, task_id)
    return deepcopy(task) if task is not None else None


def list_tasks(
    *,
    task_type: str | None = None,
    state: TaskState | None = None,
    limit: int = 100,
) -> list[TaskRecord]:
    _ensure_initialized()
    clauses: list[str] = []
    params: list[Any] = []
    if task_type:
        clauses.append("task_type = ?")
        params.append(task_type)
    if state:
        clauses.append("state = ?")
        params.append(state)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"SELECT * FROM tasks {where} ORDER BY created_index DESC LIMIT ?"
    params.append(limit)
    with _connect() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
    return [deepcopy(_row_to_task(row)) for row in rows]


def cancel_task(task_id: str) -> TaskRecord | None:
    task = get_task(task_id)
    if task is None:
        return None
    if task["state"] in {"succeeded", "failed", "cancelled", "interrupted"}:
        return task

    _update_task(
        task_id,
        cancellation_requested=True,
        event={
            "event_type": "cancellation_requested",
            "message": "task cancellation requested",
            "details": {},
            "created_at": _now_iso(),
        },
    )
    updated = get_task(task_id)
    if updated is not None and updated["state"] == "pending":
        _update_task(
            task_id,
            state="cancelled",
            error=f"task cancelled: {task_id}",
            result=None,
            finished_at=_now_iso(),
            event={
                "event_type": "cancelled",
                "message": f"task cancelled: {task_id}",
                "details": {},
                "created_at": _now_iso(),
            },
        )
        updated = get_task(task_id)
    return updated


def reset_runtime_state(*, clear_persistent_store: bool = False) -> None:
    global _INITIALIZED
    with _ACTIVE_THREADS_GUARD:
        active_threads = list(_ACTIVE_THREADS.values())
    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2.0)
    with _ACTIVE_THREADS_GUARD:
        _ACTIVE_THREADS.clear()
    with _CALLBACK_URL_LOCKS_GUARD:
        _CALLBACK_URL_LOCKS.clear()
    _INITIALIZED = False
    if clear_persistent_store:
        db_path = _db_path()
        if db_path.exists():
            db_path.unlink()
