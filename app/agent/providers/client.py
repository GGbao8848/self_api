import json
import re
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
    conversation_context: str | None = None,
) -> LLMToolDecision:
    if provider.provider == "ollama":
        return _request_ollama_decision(
            settings=settings,
            provider=provider,
            message=message,
            tools=tools,
            conversation_context=conversation_context,
        )
    if provider.provider in {"openai", "openrouter"}:
        return _request_openai_compatible_decision(
            settings=settings,
            provider=provider,
            message=message,
            tools=tools,
            conversation_context=conversation_context,
        )
    raise ProviderCallError(f"unsupported provider runtime: {provider.provider}")


def _request_ollama_decision(
    *,
    settings: Settings,
    provider: ProviderSelection,
    message: str,
    tools: list[ToolSpec],
    conversation_context: str | None = None,
) -> LLMToolDecision:
    messages = [{"role": "system", "content": _build_system_prompt(tools)}]
    if conversation_context:
        messages.append({"role": "system", "content": conversation_context})
    messages.append({"role": "user", "content": message})
    payload = {
        "model": provider.model,
        "stream": False,
        "format": "json",
        "messages": messages,
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
    conversation_context: str | None = None,
) -> LLMToolDecision:
    if provider.provider == "openai":
        base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        api_key = settings.openai_api_key or ""
    else:
        base_url = settings.openrouter_base_url.rstrip("/")
        api_key = settings.openrouter_api_key or ""

    messages = [{"role": "system", "content": _build_system_prompt(tools)}]
    if conversation_context:
        messages.append({"role": "system", "content": conversation_context})
    messages.append({"role": "user", "content": message})
    payload = {
        "model": provider.model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{base_url}/chat/completions"
    try:
        data = _post_json(
            url,
            payload,
            timeout_seconds=settings.llm_request_timeout_seconds,
            extra_headers=headers,
        )
    except ProviderCallError as exc:
        if not _supports_json_mode_fallback(str(exc)):
            raise
        payload = {
            "model": provider.model,
            "messages": messages,
            "temperature": 0,
        }
        data = _post_json(
            url,
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
        repaired = _repair_common_json_object_errors(content)
        if repaired is None:
            raise ProviderCallError(f"model returned invalid JSON decision: {content}") from exc
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError as repaired_exc:
            raise ProviderCallError(f"model returned invalid JSON decision: {content}") from repaired_exc
    if not isinstance(payload, dict):
        raise ProviderCallError("model decision must be a JSON object")

    action = str(payload.get("action") or "").strip().lower()
    message = payload.get("message")
    plan_summary = payload.get("plan_summary")
    tool_name = payload.get("tool_name")
    tool_arguments = payload.get("tool_arguments") or {}
    if not isinstance(tool_arguments, dict):
        raise ProviderCallError("tool_arguments must be an object")
    return LLMToolDecision(
        action=action,
        message=message if isinstance(message, str) else None,
        plan_summary=plan_summary if isinstance(plan_summary, str) else None,
        tool_name=tool_name if isinstance(tool_name, str) else None,
        tool_arguments=tool_arguments,
    )


def _supports_json_mode_fallback(error_message: str) -> bool:
    lowered = error_message.lower()
    return "json mode is not supported" in lowered or "response_format" in lowered


def _repair_common_json_object_errors(content: str) -> str | None:
    text = content.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None

    malformed_pair_pattern = re.compile(
        r'^\{\s*"action"\s*,\s*"(?P<action>[^"]+)"\s*,\s*'
        r'"message"\s*,\s*"(?P<message>[^"]*)"\s*,\s*'
        r'"plan_summary"\s*,\s*"(?P<plan_summary>[^"]*)"\s*,\s*'
        r'"tool_name"\s*,\s*"(?P<tool_name>[^"]*)"\s*,\s*'
        r'"tool_arguments"\s*[: ,]+\s*(?P<tool_arguments>\{.*\})\s*\}$',
        flags=re.DOTALL,
    )
    malformed_match = malformed_pair_pattern.match(text)
    if malformed_match is not None:
        repaired_payload = {
            "action": malformed_match.group("action").strip(),
            "message": malformed_match.group("message").strip(),
            "plan_summary": malformed_match.group("plan_summary").strip(),
            "tool_name": malformed_match.group("tool_name").strip(),
            "tool_arguments": json.loads(malformed_match.group("tool_arguments")),
        }
        return json.dumps(repaired_payload, ensure_ascii=False)

    # Another common case: object keys are followed by commas instead of colons.
    candidate = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*,', r'"\1":', text)
    return candidate if candidate != text else None


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
        "You are a long-running tool-using agent for dataset preprocessing.\n"
        "Your job is to advance the user's end goal one step at a time.\n"
        "Choose at most one tool for the current decision.\n"
        "Return only valid JSON with this schema:\n"
        '{"action":"execute|respond|clarify","message":"string","plan_summary":"string|null","tool_name":"string|null","tool_arguments":{}}\n'
        "Rules:\n"
        "1. Think in multi-step workflows. If the user's goal is not yet satisfied and a next tool is obvious, keep using action=execute on the next step instead of stopping early.\n"
        "2. Use action=execute only when one listed tool clearly matches the current next step.\n"
        "3. If a required path or parameter is missing and cannot be inferred from context, use action=clarify and ask for it.\n"
        "4. Use action=respond only when the user's goal is complete, or when they are chatting or asking capabilities.\n"
        "5. Never invent tool names beyond this list.\n"
        "6. For split requests like 8:2, use mode=train_val and normalized ratios.\n"
        "7. Use exact argument keys expected by the tool.\n"
        "8. Use input_dir for dataset or labels paths.\n"
        "9. For split-yolo-dataset, include output_dir as <input_dir>_split unless user gives one.\n"
        "10. For yolo-sliding-window-crop, include output_dir as <input_dir>_yolo-sliding-window-crop unless user gives one.\n"
        "11. For yolo-augment, include output_dir as <input_dir>_aug unless user gives one.\n"
        "12. For annotate-visualize, include output_dir as <input_dir>_visualized unless user gives one.\n"
        "13. For clean-nested-dataset-flat, include output_dir as <input_dir>_cleaned_flat and default flatten=true, include_backgrounds=false, pairing_mode=images_xmls_subfolders, recursive=true, copy_files=true, overwrite=true unless user gives overrides.\n"
        "14. For publish-yolo-dataset, prefer {local_paths, remote_target, classes, use_index_as_class_names}. local_paths must be an array when the user gives one or more dataset folders. If remote_target is present, it should include detector_name at the end, like sftp://host/root/detector_name. If the user explicitly says to submit directly without class names, set use_index_as_class_names=true.\n"
        "15. For publish-incremental-yolo-dataset, use {last_yaml, local_paths}; if the user gives one local dataset path, local_paths should contain exactly one item.\n"
        "16. For build-yolo-yaml, include output_yaml_path as <input_dir>/data.yaml unless project_root_dir and detector_name are both provided.\n"
        "17. For aggregate-nested-dataset, include output_dir as <input_dir>_dataset unless user gives one.\n"
        "18. For zip-folder, if output_zip_path is omitted, use <input_dir>.zip.\n"
        "19. For unzip-archive, use archive_path and include output_dir only when the user gives one or a custom extraction location is needed.\n"
        "20. For move-path and copy-path, use source_path and target_dir exactly; do not rewrite them as input_dir.\n"
        "21. For reset-yolo-label-index, use input_dir pointing to labels/, a dataset root, or a parent directory containing nested labels directories.\n"
        "22. For voc-bar-crop, include output_dir as <input_dir>_voc-bar-crop unless user gives one.\n"
        "23. For restore-voc-crops-batch, required keys are original_images_dir, original_xmls_dir, edited_crops_images_dir, edited_crops_xmls_dir, and optionally output_dir.\n"
        "24. If the current user message omits a path, but recent session context provides reusable paths, prefer the most recent output_dir or labels_dir; otherwise reuse the most recent input_dir.\n"
        "25. Read the current run progress carefully. Do not repeat the same tool with the same arguments unless the context explicitly indicates a retry.\n"
        "26. When a prior step produced labels_dir or output_dir that naturally feeds the next step, use that derived path instead of asking again.\n"
        "Available tools:\n"
        f"{tools_text}\n"
        "Argument hints:\n"
        f"{argument_text}"
    )
