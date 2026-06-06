from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model_fast: str = "gpt-4o-mini"
    openai_chat_model_smart: str = "gpt-4o"

    # Supabase / DB
    supabase_url: str
    supabase_service_key: str
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Cal.com
    calcom_api_key: str = ""
    calcom_event_type_id: int = 0
    calcom_username: str = ""

    # Vapi
    vapi_api_key: str = ""
    vapi_phone_number_id: str = ""

    # CORS — space-separated list of allowed origins
    # e.g. "https://your-frontend.vercel.app https://your-frontend.onrender.com"
    allowed_origins: str = "http://localhost:3000"


settings = Settings()
