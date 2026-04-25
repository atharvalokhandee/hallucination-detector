"""
Centralised configuration loaded from .env.
Import `settings` anywhere — never read os.environ directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Keys
    groq_api_key: str
    tavily_api_key: str

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Pipeline thresholds
    similarity_threshold_high: float = 0.75
    similarity_threshold_low: float = 0.40

    # Cache
    cache_ttl_seconds: int = 3600

    # Tavily
    tavily_max_results: int = 5

    # Groq
    groq_model: str = "llama-3.3-70b-versatile"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance — loaded once, cached forever."""
    return Settings()


# Convenience alias used throughout the codebase
settings = get_settings()
