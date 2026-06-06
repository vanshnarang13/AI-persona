# Retrieval pipeline tuning — change here, not in .env
class RetrievalConfig:
    # Tuned via scripts/tune_retrieval.py against evaluation/golden_set/questions.json
    TOP_K_CHAT: int = 8
    TOP_K_VOICE: int = 3        # leaner for < 2s voice latency
    CANDIDATE_MULTIPLIER: int = 2

    # RRF fusion weights (must sum to 1.0)
    RRF_K: int = 60
    VECTOR_WEIGHT: float = 0.55
    BM25_WEIGHT: float = 0.45

    CONFIDENCE_THRESHOLD: float = 0.35  # below this → grounding fallback
    HNSW_EF_SEARCH: int = 100


# App / server config
class AppConfig:
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    @property
    def CORS_ORIGINS(self) -> list[str]:
        from src.config.settings import settings
        return [o.strip() for o in settings.allowed_origins.split() if o.strip()]

    # Cal.com slot duration (minutes) — set by the event type, not the screener
    CALCOM_SLOT_DURATION_MINUTES: int = 30


retrieval = RetrievalConfig()
app = AppConfig()
