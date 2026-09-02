from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    model_provider: str = Field(default="openai", alias="MODEL_PROVIDER")

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")

    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        alias="DEEPSEEK_BASE_URL",
    )
    deepseek_model: str = Field(
        default="deepseek-v4-flash",
        alias="DEEPSEEK_MODEL",
    )

    doubao_api_key: Optional[str] = Field(default=None, alias="DOUBAO_API_KEY")
    doubao_base_url: str = Field(default="https://doubao-gateway.example.com/v1", alias="DOUBAO_BASE_URL")
    doubao_model: str = Field(default="doubao-seed-1-6", alias="DOUBAO_MODEL")

    glm_api_key: Optional[str] = Field(default=None, alias="GLM_API_KEY")
    glm_base_url: str = Field(default="https://glm-gateway.example.com/v1", alias="GLM_BASE_URL")
    glm_model: str = Field(default="glm-4-plus", alias="GLM_MODEL")

    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(default="https://gemini-gateway.example.com/v1", alias="GEMINI_BASE_URL")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    custom_api_key: Optional[str] = Field(default=None, alias="CUSTOM_API_KEY")
    custom_base_url: str = Field(default="https://llm-gateway.example.com/v1", alias="CUSTOM_BASE_URL")
    custom_model: str = Field(default="", alias="CUSTOM_MODEL")

    apiyi_api_key: Optional[str] = Field(default=None, alias="APIYI_KEY")
    apiyi_base_url: str = Field(
        default="https://api.apiyi.com/v1",
        alias="APIYI_BASE_URL",
    )
    apiyi_model: str = Field(
        default="deepseek-v4-flash",
        alias="APIYI_MODEL",
    )
    apiyi_gemini_model: str = Field(
        default="gemini-3.1-flash-lite",
        alias="APIYI_GEMINI_MODEL",
    )

    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173", alias="CORS_ORIGINS")
    llm_temperature: float = Field(default=0.35, alias="LLM_TEMPERATURE")
    llm_timeout_seconds: int = Field(default=300, ge=1, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=1, ge=0, le=5, alias="LLM_MAX_RETRIES")
    llm_stream_responses: bool = Field(
        default=True,
        alias="LLM_STREAM_RESPONSES",
    )

    mars_trace_archive: bool = Field(default=False, alias="MARS_TRACE_ARCHIVE")
    mars_trace_dir: str = Field(
        default="tmp/mars-traces",
        alias="MARS_TRACE_DIR",
    )
    mars_template_dir: str = Field(
        default="tmp/mars-templates",
        alias="MARS_TEMPLATE_DIR",
    )
    authoring_assistant_web_search: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "authoring_assistant_web_search",
            "AUTHORING_ASSISTANT_WEB_SEARCH",
        ),
    )
    authoring_assistant_search_timeout_seconds: int = Field(
        default=4,
        ge=1,
        le=20,
        validation_alias=AliasChoices(
            "authoring_assistant_search_timeout_seconds",
            "AUTHORING_ASSISTANT_SEARCH_TIMEOUT_SECONDS",
        ),
    )
    authoring_assistant_model_timeout_seconds: int = Field(
        default=35,
        ge=5,
        le=120,
        validation_alias=AliasChoices(
            "authoring_assistant_model_timeout_seconds",
            "AUTHORING_ASSISTANT_MODEL_TIMEOUT_SECONDS",
        ),
    )

    def cors_origin_list(self) -> List[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

    def current_llm(self) -> dict:
        provider = self.model_provider.lower().strip()
        if provider == "deepseek":
            return {
                "provider": provider,
                "api_key": self.deepseek_api_key,
                "base_url": self.deepseek_base_url,
                "model": self.deepseek_model,
            }
        if provider == "doubao":
            return {"provider": provider, "api_key": self.doubao_api_key, "base_url": self.doubao_base_url, "model": self.doubao_model}
        if provider == "glm":
            return {"provider": provider, "api_key": self.glm_api_key, "base_url": self.glm_base_url, "model": self.glm_model}
        if provider == "gemini":
            return {"provider": provider, "api_key": self.gemini_api_key, "base_url": self.gemini_base_url, "model": self.gemini_model}
        if provider == "custom":
            return {"provider": provider, "api_key": self.custom_api_key, "base_url": self.custom_base_url, "model": self.custom_model}
        if provider == "apiyi":
            return {
                "provider": provider,
                "api_key": self.apiyi_api_key,
                "base_url": self.apiyi_base_url,
                "model": self.apiyi_model,
            }
        return {"provider": "openai", "api_key": self.openai_api_key, "base_url": self.openai_base_url, "model": self.openai_model}

    def public_llm(self) -> dict[str, str | bool]:
        """Return provider status without exposing credentials or private endpoints."""
        config = self.current_llm()
        return {
            "provider": str(config["provider"]),
            "model": str(config["model"]),
            "configured": bool(config.get("api_key")),
        }

    def apiyi_authoring_assistant_config(
        self,
        model: str,
    ) -> dict[str, str | None]:
        """Resolve an allow-listed Authoring Assistant model via APIYI."""
        if model not in {
            "deepseek-v4-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.1-flash",
        }:
            raise ValueError(
                f"unsupported Authoring Assistant model: {model}"
            )
        resolved = self.apiyi_model
        if model != "deepseek-v4-flash":
            resolved = self.apiyi_gemini_model
            # APIYI does not expose a text model under the historical
            # gemini-3.1-flash identifier. Migrate old .env files safely.
            if resolved == "gemini-3.1-flash":
                resolved = "gemini-3.1-flash-lite"
        return {
            "provider": "apiyi",
            "api_key": self.apiyi_api_key,
            "base_url": self.apiyi_base_url,
            "model": resolved,
        }

@lru_cache
def get_settings() -> Settings:
    return Settings()
