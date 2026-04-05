from __future__ import annotations

from abc import ABC, abstractmethod

from causal_inference.settings import Settings


class LLMClient(ABC):
    """Abstract base for LLM chat providers."""

    @abstractmethod
    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIChatClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()


class AnthropicClaudeClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        chunks: list[str] = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                chunks.append(getattr(block, "text", ""))
        return "".join(chunks).strip()


def build_llm_client(settings: Settings) -> LLMClient:
    """Factory: return an LLMClient based on the configured provider."""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAIChatClient(
            api_key=settings.openai_api_key, model=settings.openai_model
        )
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic"
            )
        return AnthropicClaudeClient(
            api_key=settings.anthropic_api_key, model=settings.anthropic_model
        )
    raise ValueError("Unsupported LLM_PROVIDER. Use 'openai' or 'anthropic'.")
