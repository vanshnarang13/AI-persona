from __future__ import annotations
import asyncpg
from supabase import create_client, Client
from src.config import settings

_pool: asyncpg.Pool | None = None
_supabase: Client | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        # asyncpg needs postgresql://, not the SQLAlchemy postgresql+asyncpg:// form
        dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
        _pool = await asyncpg.create_pool(
            dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
            statement_cache_size=0,  # required for Supabase transaction pooler
        )
    return _pool


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(settings.supabase_url, settings.supabase_service_key)
    return _supabase


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
