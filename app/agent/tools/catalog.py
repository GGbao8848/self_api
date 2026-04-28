from app.agent.tools.registry import get_tool_specs
from app.agent.types import ToolSpec


_EXECUTABLE_TOOL_NAMES = {
    "scan-yolo-label-indices",
    "rewrite-yolo-label-indices",
    "xml-to-yolo",
    "split-yolo-dataset",
}


def get_executable_tool_specs() -> list[ToolSpec]:
    return [tool for tool in get_tool_specs() if tool.name in _EXECUTABLE_TOOL_NAMES]
