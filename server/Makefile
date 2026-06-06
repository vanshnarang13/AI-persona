.PHONY: start stop restart logs-api db-migrate ingest-all ingest-resume verify-retrieval tune-retrieval eval dev

start:
	docker compose up -d

stop:
	docker compose down

restart:
	docker compose restart api

logs-api:
	docker compose logs -f api

# ── database ──────────────────────────────────────────────────────
db-migrate:
	@echo "Running migration against Supabase..."
	cd server && python -c "\
import asyncio, asyncpg, os; from dotenv import load_dotenv; load_dotenv('.env'); \
dsn = os.environ['DATABASE_URL'].replace('postgresql+asyncpg://', 'postgresql://'); \
sql = open('supabase/migrations/0001_init.sql').read(); \
asyncio.run(asyncpg.connect(dsn, statement_cache_size=0).then(lambda c: c.execute(sql)))" \
	|| psql "$(shell grep DATABASE_URL .env | cut -d= -f2- | sed 's/postgresql+asyncpg/postgresql/')" \
	     -f supabase/migrations/0001_init.sql

# ── ingestion ─────────────────────────────────────────────────────
ingest-all:
	PYTHONPATH=. python ingest/ingest_corpus.py

ingest-resume:
	PYTHONPATH=. python ingest/ingest_resume.py

verify-retrieval:
	PYTHONPATH=. python scripts/verify_retrieval_basic.py

tune-retrieval:
	PYTHONPATH=. python scripts/tune_retrieval.py

# ── evals ─────────────────────────────────────────────────────────
eval:
	PYTHONPATH=. python evaluation/run_evals.py

# ── local dev (no docker) ─────────────────────────────────────────
dev:
	PYTHONPATH=. uvicorn src.server:app --reload --port 8000
