from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuration for LLM provider selection and API keys."""

    llm_provider: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    openai_model: str
    anthropic_model: str

    @classmethod
    def from_env(cls, env_path: Path | str | None = None) -> Settings:
        if env_path is not None:
            load_dotenv(env_path)
        else:
            for candidate in [
                Path.cwd() / ".env",
                Path(__file__).resolve().parents[2] / ".env",
            ]:
                if candidate.exists():
                    load_dotenv(candidate)
                    break

        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        )
