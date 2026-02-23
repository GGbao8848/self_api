import logging
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from json import dumps
from typing import Any, Callable, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TaskState = Literal["pending", "running", "succeeded", "failed"]
CallbackState = Literal["pending", "running", "succeeded", "failed"]
_IDEMPOTENT_CALLBACK_SUCCESS_CODES = {409}
_UNSET = object()


class TaskRecord(TypedDict):
    task_id: str
    task_type: str
    state: TaskState
    created_at: str
    updated_at: str
    result: dict[str, Any] | None
    error: str | None
    callback_url: str | None
    callback_state: CallbackState
    callback_sent_at: str | None
    callback_status_code: int | None
    callback_error: str | None
    callback_events: list[dict[str, Any]]


_TASKS: dict[str, TaskRecord] = {}
_LOCK = threading.Lock()
_CALLBACK_URL_LOCKS: dict[str, threading.Lock] = {}
_CALLBACK_URL_LOCKS_GUARD = threading.Lock()
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_task(
    task_id: str,
    *,
    state: TaskState | None = None,
    result: dict[str, Any] | None | object = _UNSET,
    error: str | None | object = _UNSET,
    callback_state: CallbackState | None = None,
    callback_sent_at: str | None | object = _UNSET,
    callback_status_code: int | None | object = _UNSET,
    callback_error: str | None | object = _UNSET,
    callback_event: dict[str, Any] | object = _UNSET,
) -> None:
    with _LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            return
        if state is not None:
            task["state"] = state
        if result is not _UNSET:
            task["result"] = result
        if error is not _UNSET:
            task["error"] = error
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
        task["updated_at"] = _now_iso()


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
        "result": task["result"],
        "error": task["error"],
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
                logger.warning(
                    "webhook callback sent: task_id=%s state=%s callback_url=%s method=%s status_code=%s success=false error=%s",
                    task_id,
                    event_state,
                    task["callback_url"],
                    method_used,
                    status_code,
                    f"callback returned non-2xx status: {status_code}",
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
                logger.info(
                    "webhook callback sent: task_id=%s state=%s callback_url=%s method=%s status_code=%s success=true",
                    task_id,
                    event_state,
                    task["callback_url"],
                    method_used,
                    exc.code,
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
                logger.warning(
                    "webhook callback sent: task_id=%s state=%s callback_url=%s method=%s status_code=%s success=false error=%s",
                    task_id,
                    event_state,
                    task["callback_url"],
                    method_used,
                    exc.code,
                    f"callback returned non-2xx status: {exc.code}",
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
            logger.warning(
                "webhook callback sent: task_id=%s state=%s callback_url=%s method=%s status_code=%s success=false error=%s",
                task_id,
                event_state,
                task["callback_url"],
                method_used,
                "none",
                str(exc),
            )


def submit_task(
    task_type: str,
    runner: Callable[[], dict[str, Any]],
    *,
    callback_url: str | None = None,
    callback_timeout_seconds: float = 10.0,
) -> str:
    task_id = uuid.uuid4().hex
    now = _now_iso()
    # If callback is not requested, treat callback lifecycle as already completed.
    callback_state: CallbackState = "pending" if callback_url else "succeeded"
    with _LOCK:
        _TASKS[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "state": "pending",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
            "callback_url": callback_url,
            "callback_state": callback_state,
            "callback_sent_at": None,
            "callback_status_code": None,
            "callback_error": None,
            "callback_events": [],
        }

    def _run() -> None:
        _update_task(task_id, state="running")
        try:
            task_result = runner()
        except Exception as exc:  # noqa: BLE001
            _update_task(task_id, state="failed", error=str(exc), result=None)
            if callback_url:
                _send_task_callback(task_id, callback_timeout_seconds)
        else:
            _update_task(task_id, state="succeeded", result=task_result, error=None)
            if callback_url:
                _send_task_callback(task_id, callback_timeout_seconds)
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
