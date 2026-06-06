from __future__ import annotations
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.config.tuning import app as app_config
from src.services.supabase_client import get_pool, close_pool
from src.services.redis_client import get_redis_binary, close_redis
from src.rag.retrieval import get_or_build_index
from src.middleware import StructlogMiddleware, RateLimitMiddleware
from src.routes import health_router, retrieve_router, chat_router, booking_router, vapi_tools_router
from src.agents.persona_agent.graph import build_persona_graph
from src.agents.booking_agent.graph import build_booking_graph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import psycopg

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("server.startup")

    # DB pool
    app.state.pool = await get_pool()
    log.info("server.db_pool_ready")

    # BM25 index (Redis-cached)
    app.state.bm25 = await get_or_build_index(app.state.pool, get_redis_binary())
    log.info("server.bm25_index_ready")

    # Persona graph (no checkpointing — stateless per request)
    app.state.persona_graph = build_persona_graph(
        pool=app.state.pool,
        bm25_index=app.state.bm25,
        redis=get_redis_binary(),
    )
    log.info("server.persona_graph_ready")

    # Booking graph — AsyncPostgresSaver persists booking state in Supabase Postgres.
    # Survives restarts and works across multiple workers.
    pg_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    # setup() runs CREATE INDEX CONCURRENTLY which requires autocommit — one-off connection
    async with await psycopg.AsyncConnection.connect(pg_url, autocommit=True) as setup_conn:
        await AsyncPostgresSaver(setup_conn).setup()
    # Pool for all runtime checkpointing
    app.state.pg_pool = AsyncConnectionPool(conninfo=pg_url, min_size=1, max_size=5, open=False)
    await app.state.pg_pool.open()
    checkpointer = AsyncPostgresSaver(app.state.pg_pool)
    app.state.booking_graph = build_booking_graph(checkpointer)
    log.info("server.booking_graph_ready")

    # Pre-warm retrieval cache — ensures first voice call hits cache (~300ms) not cold (~2.5s)
    import asyncio
    from src.rag.retrieval import retrieve as _retrieve

    async def _warmup() -> None:
        WARMUP_QUERIES = [
            "background education IIT Roorkee",
            "AI machine learning experience projects",
            "skills programming languages frameworks",
            "AtlasRAG TradeSmith SteganoGAN projects",
            "work internship Noos Technologies experience",
            "why hire Vansh fit for role Scaler",
        ]
        redis = get_redis_binary()
        for q in WARMUP_QUERIES:
            try:
                await _retrieve(q, "voice", app.state.pool, app.state.bm25, redis)
            except Exception as exc:
                log.warning("server.warmup_query_failed", query=q, error=str(exc))
        log.info("server.cache_warmup_done", queries=len(WARMUP_QUERIES))

    asyncio.create_task(_warmup())
    log.info("server.cache_warmup_started")

    yield

    # Graceful shutdown
    await close_pool()
    await close_redis()
    await app.state.pg_pool.close()
    log.info("server.shutdown")


app = FastAPI(
    title="Persona Agent API",
    description="AI persona of Vansh Narang — recruiter chat + voice + booking",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(StructlogMiddleware)

app.include_router(health_router)
app.include_router(retrieve_router)
app.include_router(chat_router)
app.include_router(booking_router)
app.include_router(vapi_tools_router)
