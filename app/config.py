from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    groq_api_key: str
    tavily_api_key: str
    redis_url: str = "redis://localhost:6379"
    similarity_threshold_high: float = 0.75
    similarity_threshold_low: float = 0.40
    cache_ttl_seconds: int = 3600
    tavily_max_results: int = 5
    groq_model: str = "llama-3.3-70b-versatile"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()