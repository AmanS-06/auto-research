from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    database_url: str = ""
    database_url_sync: str = ""
    database_pool_size: int = 10
    database_max_overflow: int = 20

    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_timeout_seconds: float = 60.0

    serper_api_key: str = ""
    serper_base_url: str = "https://google.serper.dev"
    serper_timeout_seconds: float = 20.0

    max_research_tasks: int = 5
    max_sources_per_task: int = 3
    min_source_relevance: float = 0.4


settings = Settings()
