import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.agent.types import LLMToolDecision, ProviderSelection, ToolSpec
from app.core.config import Settings


class ProviderCallError(RuntimeError):
    pass


def request_tool_decision(
    *,
    settings: Settings,
    provider: ProviderSelection,
    message: str,
    tools: list[ToolSpec],
) -> LLMToolDecision:
    if provider.provider == "ollama":
        return _request_ollama_decision(
            settings=settings,
            provider=provider,
            message=message,
            tools=tools,
        )
    if provider.provider in {"openai", "openrouter"}:
        return _request_openai_compatible_decision(
            settings=settings,
            provider=provider,
            message=message,
            tools=tools,
        )
    raise ProviderCallError(f"unsupported provider runtime: {provider.provider}")


def _request_ollama_decision(
    *,
    settings: Settings,
    provider: ProviderSelection,
    message: str,
    tools: list[ToolSpec],
) -> LLMToolDecision:
    payload = {
        "model": provider.model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": _build_system_prompt(tools)},
            {"role": "user", "content": message},
        ],
        "options": {"temperature": 0},
    }
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    data = _post_json(url, payload, timeout_seconds=settings.llm_request_timeout_seconds)
    content = (
        data.get("message", {}).get("content")
        if isinstance(data.get("message"), dict)
        else None
    )
    if not isinstance(content, str) or not content.strip():
        raise ProviderCallError("ollama returned empty message content")
    return _parse_decision_json(content)


def _request_openai_compatible_decision(
    *,
    settings: Settings,
    provider: ProviderSelection,
    message: str,
    tools: list[ToolSpec],
) -> LLMToolDecision:
    if provider.provider == "openai":
        base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        api_key = settings.openai_api_key or ""
    else:
        base_url = settings.openrouter_base_url.rstrip("/")
        api_key = settings.openrouter_api_key or ""

    payload = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": _build_system_prompt(tools)},
            {"role": "user", "content": message},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    data = _post_json(
        f"{base_url}/chat/completions",
        payload,
        timeout_seconds=settings.llm_request_timeout_seconds,
        extra_headers=headers,
    )
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderCallError("provider returned no choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ProviderCallError("provider returned empty message content")
    return _parse_decision_json(content)


def _post_json(
    url: str,
    payload: dict,
    *,
    timeout_seconds: int,
    extra_headers: dict | None = None,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise ProviderCallError(f"provider returned {exc.code}: {detail or 'unknown error'}") from exc
    except URLError as exc:
        raise ProviderCallError(f"failed to reach provider: {exc}") from exc
    except TimeoutError as exc:
        raise ProviderCallError(f"provider request timed out after {timeout_seconds}s") from exc

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderCallError("provider returned invalid JSON payload") from exc
    if not isinstance(parsed, dict):
        raise ProviderCallError("provider returned non-object JSON payload")
    return parsed


def _parse_decision_json(content: str) -> LLMToolDecision:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProviderCallError(f"model returned invalid JSON decision: {content}") from exc
    if not isinstance(payload, dict):
        raise ProviderCallError("model decision must be a JSON object")

    action = str(payload.get("action") or "").strip().lower()
    message = payload.get("message")
    tool_name = payload.get("tool_name")
    tool_arguments = payload.get("tool_arguments") or {}
    if not isinstance(tool_arguments, dict):
        raise ProviderCallError("tool_arguments must be an object")
    return LLMToolDecision(
        action=action,
        message=message if isinstance(message, str) else None,
        tool_name=tool_name if isinstance(tool_name, str) else None,
        tool_arguments=tool_arguments,
    )


def _build_system_prompt(tools: list[ToolSpec]) -> str:
    tool_lines = [
        f"- {tool.name}: {tool.description} (async_task={str(tool.async_task).lower()})"
        for tool in tools
    ]
    tools_text = "\n".join(tool_lines)
    argument_lines = [
        f"- {tool.name}: {tool.argument_hint}"
        for tool in tools
        if tool.argument_hint
    ]
    argument_text = "\n".join(argument_lines)
    return (
        "You are a tool-selection agent for dataset preprocessing.\n"
        "Choose at most one tool.\n"
        "Return only valid JSON with this schema:\n"
        '{"action":"execute|respond|clarify","message":"string","tool_name":"string|null","tool_arguments":{}}\n'
        "Rules:\n"
        "1. Use action=execute only when one listed tool clearly matches.\n"
        "2. If a required path or parameter is missing, use action=clarify and ask for it.\n"
        "3. If the user is chatting or asking capabilities, use action=respond.\n"
        "4. Never invent tool names beyond this list.\n"
        "5. For split requests like 8:2, use mode=train_val and normalized ratios.\n"
        "6. Use exact argument keys expected by the tool.\n"
        "7. Use input_dir for dataset or labels paths.\n"
        "8. For split-yolo-dataset, include output_dir as <input_dir>_split unless user gives one.\n"
        "9. For yolo-sliding-window-crop, include output_dir as <input_dir>_yolo-sliding-window-crop unless user gives one.\n"
        "10. For yolo-augment, include output_dir as <input_dir>_aug unless user gives one.\n"
        "11. For annotate-visualize, include output_dir as <input_dir>_visualized unless user gives one.\n"
        "12. For clean-nested-dataset-flat, include output_dir as <input_dir>_cleaned_flat and default flatten=true, include_backgrounds=false, pairing_mode=images_xmls_subfolders, recursive=true, copy_files=true, overwrite=true unless user gives overrides.\n"
        "13. For publish-incremental-yolo-dataset, use {last_yaml, local_paths}; if the user gives one local dataset path, local_paths should contain exactly one item.\n"
        "14. For build-yolo-yaml, include output_yaml_path as <input_dir>/data.yaml unless project_root_dir and detector_name are both provided.\n"
        "Available tools:\n"
        f"{tools_text}\n"
        "Argument hints:\n"
        f"{argument_text}"
    )
