from app.agent.providers.client import (
    ProviderCallError,
    _parse_decision_json,
    _request_openai_compatible_decision,
)
from app.agent.types import ProviderSelection, ToolSpec
from app.core.config import Settings


def test_parse_decision_json_repairs_common_malformed_object() -> None:
    content = (
        '{"action","execute","message","Executing xml-to-yolo conversion for /tmp/dataset",'
        '"plan_summary","Convert XML annotations to YOLO format",'
        '"tool_name","xml-to-yolo","tool_arguments" : {"input_dir" : " /tmp/dataset" } }'
    )

    decision = _parse_decision_json(content)

    assert decision.action == "execute"
    assert decision.tool_name == "xml-to-yolo"
    assert decision.tool_arguments == {"input_dir": " /tmp/dataset"}


def test_openai_compatible_decision_falls_back_when_json_mode_unsupported(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int, extra_headers: dict | None = None) -> dict:
        calls.append(payload)
        if len(calls) == 1:
            raise ProviderCallError(
                'provider returned 400: {"error":{"message":"Provider returned error","code":400,'
                '"metadata":{"raw":"{\\"code\\":20024,\\"message\\":\\"Json mode is not supported for this model.\\",'
                '\\"data\\":null}"}}}'
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"action":"respond","message":"处理完成","plan_summary":"done","tool_name":null,"tool_arguments":{}}'
                    }
                }
            ]
        }

    monkeypatch.setattr("app.agent.providers.client._post_json", fake_post_json)
    settings = Settings(
        openrouter_api_key="test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
    )
    provider = ProviderSelection(
        provider="openrouter",
        model="openai/gpt-oss-20b:free",
        configured=True,
    )

    decision = _request_openai_compatible_decision(
        settings=settings,
        provider=provider,
        message="你好",
        tools=[ToolSpec(name="xml-to-yolo", description="x", async_task=True)],
    )

    assert decision.action == "respond"
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
