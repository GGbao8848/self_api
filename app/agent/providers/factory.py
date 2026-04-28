from app.agent.types import ProviderSelection
from app.core.config import Settings


_KNOWN_PROVIDERS = {"openai", "openrouter", "ollama"}


def select_provider(
    settings: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> ProviderSelection:
    selected_provider = (provider or settings.llm_default_provider or "").strip().lower()
    selected_model = (model or settings.llm_default_model or "").strip() or None

    if not selected_provider:
        return ProviderSelection(
            provider="",
            model=selected_model,
            configured=False,
            reason="SELF_API_LLM_DEFAULT_PROVIDER is not configured",
        )

    if selected_provider not in _KNOWN_PROVIDERS:
        return ProviderSelection(
            provider=selected_provider,
            model=selected_model,
            configured=False,
            reason=f"unsupported provider: {selected_provider}",
        )

    if selected_provider == "openai" and not settings.openai_api_key:
        return ProviderSelection(
            provider=selected_provider,
            model=selected_model,
            configured=False,
            reason="SELF_API_OPENAI_API_KEY is not configured",
        )

    if selected_provider == "openrouter" and not settings.openrouter_api_key:
        return ProviderSelection(
            provider=selected_provider,
            model=selected_model,
            configured=False,
            reason="SELF_API_OPENROUTER_API_KEY is not configured",
        )

    if selected_provider == "ollama":
        selected_model = selected_model or settings.ollama_model
        if not selected_model:
            return ProviderSelection(
                provider=selected_provider,
                model=None,
                configured=False,
                reason="SELF_API_OLLAMA_MODEL or SELF_API_LLM_DEFAULT_MODEL is not configured",
            )

    return ProviderSelection(
        provider=selected_provider,
        model=selected_model,
        configured=True,
    )
