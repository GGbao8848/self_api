from app.agent.tools.registry import get_tool_definitions
from app.agent.types import ToolSpec


def get_executable_tool_specs() -> list[ToolSpec]:
    return [tool.to_spec() for tool in get_tool_definitions()]
