import time
from collections.abc import Callable

from pydantic import BaseModel

from app.services.task_manager import get_task, submit_task


def submit_and_wait_for_task(
    *,
    task_type: str,
    payload: BaseModel,
    runner: Callable[[], dict],
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 0.2,
) -> dict:
    task_id = submit_task(
        task_type=task_type,
        runner=runner,
    )
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        task = get_task(task_id)
        if task is None:
            return {
                "task_id": task_id,
                "task_type": task_type,
                "state": "failed",
                "error": f"task not found: {task_id}",
            }
        if task["state"] in {"succeeded", "failed", "cancelled"}:
            return _compact_task_result(task)
        time.sleep(poll_interval_seconds)

    return {
        "task_id": task_id,
        "task_type": task_type,
        "state": "failed",
        "error": f"task timed out after {timeout_seconds:.1f}s",
        "request": payload.model_dump(),
    }


def _compact_task_result(task: dict) -> dict:
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
    }
