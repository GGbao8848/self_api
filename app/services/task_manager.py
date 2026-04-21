import logging
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from json import dumps
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.services.artifact_store import register_path, summarize_record

TaskState = Literal["pending", "running", "succeeded", "failed", "cancelled"]
CallbackState = Literal["pending", "running", "succeeded", "failed"]
_IDEMPOTENT_CALLBACK_SUCCESS_CODES = {409}
_UNSET = object()
_TASK_LOCAL = threading.local()


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
    queue_key: str | None
    queue_position: int | None


_TASKS: dict[str, TaskRecord] = {}
_LOCK = threading.RLock()
_QUEUE_CONDITION = threading.Condition(_LOCK)
_TASK_QUEUES: dict[str, list[str]] = {}
_ACTIVE_QUEUED_TASKS: dict[str, str] = {}
_CALLBACK_URL_LOCKS: dict[str, threading.Lock] = {}
_CALLBACK_URL_LOCKS_GUARD = threading.Lock()
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    queue_position: int | None | object = _UNSET,
) -> None:
    with _LOCK:
        task = _TASKS.get(task_id)
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
        if queue_position is not _UNSET:
            task["queue_position"] = queue_position
        task["updated_at"] = _now_iso()


def _refresh_queue_positions(queue_key: str) -> None:
    queue = _TASK_QUEUES.get(queue_key, [])
    for index, queued_task_id in enumerate(queue, start=1):
        task = _TASKS.get(queued_task_id)
        if task is not None:
            task["queue_position"] = index
            task["updated_at"] = _now_iso()


def _enqueue_task(task_id: str, queue_key: str) -> None:
    with _QUEUE_CONDITION:
        queue = _TASK_QUEUES.setdefault(queue_key, [])
        queue.append(task_id)
        _refresh_queue_positions(queue_key)
        _QUEUE_CONDITION.notify_all()


def _acquire_task_queue_slot(task_id: str, queue_key: str) -> bool:
    with _QUEUE_CONDITION:
        while True:
            task = _TASKS.get(task_id)
            if task is None:
                return False
            if task["cancellation_requested"] or task["state"] == "cancelled":
                queue = _TASK_QUEUES.get(queue_key, [])
                if task_id in queue:
                    queue.remove(task_id)
                    _refresh_queue_positions(queue_key)
                _QUEUE_CONDITION.notify_all()
                return False

            queue = _TASK_QUEUES.get(queue_key, [])
            active_task_id = _ACTIVE_QUEUED_TASKS.get(queue_key)
            is_queue_head = bool(queue) and queue[0] == task_id
            if active_task_id is None and is_queue_head:
                _ACTIVE_QUEUED_TASKS[queue_key] = task_id
                queue.pop(0)
                task["queue_position"] = None
                task["updated_at"] = _now_iso()
                _refresh_queue_positions(queue_key)
                _QUEUE_CONDITION.notify_all()
                return True

            _QUEUE_CONDITION.wait()


def _release_task_queue_slot(task_id: str, queue_key: str) -> None:
    with _QUEUE_CONDITION:
        if _ACTIVE_QUEUED_TASKS.get(queue_key) == task_id:
            _ACTIVE_QUEUED_TASKS.pop(queue_key, None)

        queue = _TASK_QUEUES.get(queue_key, [])
        if task_id in queue:
            queue.remove(task_id)
        _refresh_queue_positions(queue_key)

        if not queue:
            _TASK_QUEUES.pop(queue_key, None)
        if queue_key not in _TASK_QUEUES:
            _ACTIVE_QUEUED_TASKS.pop(queue_key, None)

        _QUEUE_CONDITION.notify_all()


def _post_callback(callback_url: str, payload: dict[str, Any], timeout: float) -> int:
    body = dumps(payload).encode("utf-8")
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
    callback_url = task["callback_url"]
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

    callback_lock = _get_callback_url_lock(callback_url)
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

            if (200 <= status_code < 300) or (
                status_code in _IDEMPOTENT_CALLBACK_SUCCESS_CODES
            ):
                _update_task(
                    task_id,
                    callback_state="succeeded",
                    callback_sent_at=_now_iso(),
                    callback_status_code=status_code,
                    callback_error=None,
                    callback_event={
                        "state": event_state,
                        "attempted_at": event_time,
                        "callback_url": task["callback_url"],
                        "status_code": status_code,
                        "method": method_used,
                        "success": True,
                        "error": None,
                    },
                )
                logger.info(
                    "webhook callback sent: task_id=%s state=%s callback_url=%s method=%s status_code=%s success=true",
                    task_id,
                    event_state,
                    task["callback_url"],
                    method_used,
                    status_code,
                )
            else:
                _update_task(
                    task_id,
                    callback_state="failed",
                    callback_sent_at=_now_iso(),
                    callback_status_code=status_code,
                    callback_error=f"callback returned non-2xx status: {status_code}",
                    callback_event={
                        "state": event_state,
                        "attempted_at": event_time,
                        "callback_url": task["callback_url"],
                        "status_code": status_code,
                        "method": method_used,
                        "success": False,
                        "error": f"callback returned non-2xx status: {status_code}",
                    },
                )
        except HTTPError as exc:
            if exc.code in _IDEMPOTENT_CALLBACK_SUCCESS_CODES:
                _update_task(
                    task_id,
                    callback_state="succeeded",
                    callback_sent_at=_now_iso(),
                    callback_status_code=exc.code,
                    callback_error=None,
                    callback_event={
                        "state": event_state,
                        "attempted_at": event_time,
                        "callback_url": task["callback_url"],
                        "status_code": exc.code,
                        "method": method_used,
                        "success": True,
                        "error": None,
                    },
                )
            else:
                _update_task(
                    task_id,
                    callback_state="failed",
                    callback_sent_at=_now_iso(),
                    callback_status_code=exc.code,
                    callback_error=f"callback returned non-2xx status: {exc.code}",
                    callback_event={
                        "state": event_state,
                        "attempted_at": event_time,
                        "callback_url": task["callback_url"],
                        "status_code": exc.code,
                        "method": method_used,
                        "success": False,
                        "error": f"callback returned non-2xx status: {exc.code}",
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
    queue_key: str | None = None,
) -> str:
    task_id = uuid.uuid4().hex
    now = _now_iso()
    callback_state: CallbackState = "pending" if callback_url else "succeeded"
    with _LOCK:
        _TASKS[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "state": "pending",
            "created_at": now,
            "updated_at": now,
            "finished_at": None,
            "result": None,
            "error": None,
            "cancellation_requested": False,
            "callback_url": callback_url,
            "callback_state": callback_state,
            "callback_sent_at": None,
            "callback_status_code": None,
            "callback_error": None,
            "callback_events": [],
            "artifacts": [],
            "queue_key": queue_key,
            "queue_position": None,
        }
    if queue_key:
        _enqueue_task(task_id, queue_key)

    def _run() -> None:
        _set_current_task_id(task_id)
        try:
            if queue_key and not _acquire_task_queue_slot(task_id, queue_key):
                task = get_task(task_id)
                if (
                    task is not None
                    and task["state"] == "cancelled"
                    and callback_url
                ):
                    _send_task_callback(task_id, callback_timeout_seconds)
                return
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
            )
            if callback_url:
                _send_task_callback(task_id, callback_timeout_seconds)
        finally:
            if queue_key:
                _release_task_queue_slot(task_id, queue_key)
            _set_current_task_id(None)

    thread = threading.Thread(
        target=_run,
        daemon=True,
        name=f"preprocess-task-{task_id[:8]}",
    )
    thread.start()
    return task_id


def get_task(task_id: str) -> TaskRecord | None:
    with _LOCK:
        task = _TASKS.get(task_id)
        return deepcopy(task) if task is not None else None


def list_tasks(
    *,
    task_type: str | None = None,
    state: TaskState | None = None,
    limit: int = 100,
) -> list[TaskRecord]:
    with _LOCK:
        tasks = list(_TASKS.values())
    tasks.sort(key=lambda item: item["created_at"], reverse=True)
    filtered: list[TaskRecord] = []
    for task in tasks:
        if task_type and task["task_type"] != task_type:
            continue
        if state and task["state"] != state:
            continue
        filtered.append(deepcopy(task))
        if len(filtered) >= limit:
            break
    return filtered


def cancel_task(task_id: str) -> TaskRecord | None:
    task = get_task(task_id)
    if task is None:
        return None
    if task["state"] in {"succeeded", "failed", "cancelled"}:
        return task

    _update_task(task_id, cancellation_requested=True)
    updated = get_task(task_id)
    if updated is not None and updated["state"] == "pending":
        queue_key = updated.get("queue_key")
        with _QUEUE_CONDITION:
            if queue_key and task_id in _TASK_QUEUES.get(queue_key, []):
                _TASK_QUEUES[queue_key].remove(task_id)
                _refresh_queue_positions(queue_key)
                _QUEUE_CONDITION.notify_all()
            _update_task(
                task_id,
                state="cancelled",
                error=f"task cancelled: {task_id}",
                result=None,
                finished_at=_now_iso(),
                queue_position=None,
            )
        updated = get_task(task_id)
    return updated
