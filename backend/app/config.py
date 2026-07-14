from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_provider: str = Field(default="openai", alias="MODEL_PROVIDER")

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")

    doubao_api_key: Optional[str] = Field(default=None, alias="DOUBAO_API_KEY")
    doubao_base_url: str = Field(default="https://your-doubao-or-company-gateway/v1", alias="DOUBAO_BASE_URL")
    doubao_model: str = Field(default="doubao-seed-1-6", alias="DOUBAO_MODEL")

    glm_api_key: Optional[str] = Field(default=None, alias="GLM_API_KEY")
    glm_base_url: str = Field(default="https://your-glm-or-company-gateway/v1", alias="GLM_BASE_URL")
    glm_model: str = Field(default="glm-4-plus", alias="GLM_MODEL")

    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(default="https://your-gemini-compatible-gateway/v1", alias="GEMINI_BASE_URL")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    custom_api_key: Optional[str] = Field(default=None, alias="CUSTOM_API_KEY")
    custom_base_url: str = Field(default="https://your-company-gateway/v1", alias="CUSTOM_BASE_URL")
    custom_model: str = Field(default="", alias="CUSTOM_MODEL")

    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173", alias="CORS_ORIGINS")
    llm_temperature: float = Field(default=0.35, alias="LLM_TEMPERATURE")
    llm_timeout_seconds: int = Field(default=45, alias="LLM_TIMEOUT_SECONDS")

    def cors_origin_list(self) -> List[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

    def current_llm(self) -> dict:
        provider = self.model_provider.lower().strip()
        if provider == "doubao":
            return {"provider": provider, "api_key": self.doubao_api_key, "base_url": self.doubao_base_url, "model": self.doubao_model}
        if provider == "glm":
            return {"provider": provider, "api_key": self.glm_api_key, "base_url": self.glm_base_url, "model": self.glm_model}
        if provider == "gemini":
            return {"provider": provider, "api_key": self.gemini_api_key, "base_url": self.gemini_base_url, "model": self.gemini_model}
        if provider == "custom":
            return {"provider": provider, "api_key": self.custom_api_key, "base_url": self.custom_base_url, "model": self.custom_model}
        return {"provider": "openai", "api_key": self.openai_api_key, "base_url": self.openai_base_url, "model": self.openai_model}


@lru_cache
def get_settings() -> Settings:
    return Settings()
