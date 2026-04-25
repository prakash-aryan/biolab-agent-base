"""Settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from biolab_agent.agent.base import AgentConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    biolab_llm_model: str = Field("medgemma:4b")
    biolab_embed_model: str = Field("nomic-embed-text")
    biolab_lora_adapter: str | None = Field(default=None)

    biolab_device: str = Field("cuda")

    ollama_host: str = Field("http://ollama:11434")
    qdrant_url: str = Field("http://qdrant:6333")

    biolab_data_dir: Path = Field(Path("/data"))
    biolab_artifact_dir: Path = Field(Path("/artifacts"))

    log_level: str = Field("INFO")

    def to_agent_config(self) -> AgentConfig:
        return AgentConfig(
            llm_model=self.biolab_llm_model,
            embed_model=self.biolab_embed_model,
            ollama_host=self.ollama_host,
            qdrant_url=self.qdrant_url,
            data_dir=str(self.biolab_data_dir),
            artifact_dir=str(self.biolab_artifact_dir),
            device=self.biolab_device,
            lora_adapter=self.biolab_lora_adapter,
        )


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()
