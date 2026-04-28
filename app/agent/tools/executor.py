from pydantic import ValidationError

from app.agent.tools.async_runner import submit_and_wait_for_task
from app.agent.tools.registry import AgentToolDefinition, get_tool_definition
from app.agent.types import ToolCallRecord


class AgentToolError(ValueError):
    pass


def execute_tool(name: str, arguments: dict) -> ToolCallRecord:
    definition = get_tool_definition(name)
    if definition is None:
        raise AgentToolError(f"tool is not executable yet: {name}")

    normalized_arguments = _normalize_tool_arguments(definition, arguments)
    if definition.async_task:
        return _execute_async_tool(definition, normalized_arguments)
    return _execute_sync_tool(definition, normalized_arguments)


def _normalize_tool_arguments(definition: AgentToolDefinition, arguments: dict) -> dict:
    normalized = dict(arguments)
    if "input_dir" not in normalized:
        for alias in ("path", "dir", "dataset_dir", "dataset_path", "labels_dir"):
            value = normalized.pop(alias, None)
            if value:
                normalized["input_dir"] = value
                break
    if definition.normalize_arguments is not None:
        normalized = definition.normalize_arguments(normalized)
    return normalized


def _execute_async_tool(definition: AgentToolDefinition, arguments: dict) -> ToolCallRecord:
    try:
        payload = definition.request_model(**arguments)
        result = submit_and_wait_for_task(
            task_type=definition.task_type or definition.name.replace("-", "_"),
            payload=payload,
            runner=lambda: definition.runner(payload).model_dump(),
        )
    except (ValidationError, ValueError) as exc:
        return ToolCallRecord(
            name=definition.name,
            arguments=arguments,
            error=str(exc),
        )
    return ToolCallRecord(
        name=definition.name,
        arguments=payload.model_dump(),
        result=result,
        error=result.get("error"),
    )


def _execute_sync_tool(definition: AgentToolDefinition, arguments: dict) -> ToolCallRecord:
    try:
        payload = definition.request_model(**arguments)
        result = definition.runner(payload)
    except (ValidationError, ValueError) as exc:
        return ToolCallRecord(
            name=definition.name,
            arguments=arguments,
            error=str(exc),
        )
    return ToolCallRecord(
        name=definition.name,
        arguments=payload.model_dump(),
        result=result.model_dump(),
    )
