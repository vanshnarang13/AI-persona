-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Main documents table ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  content     TEXT        NOT NULL,
  embedding   VECTOR(1536),
  source      TEXT        NOT NULL,
  source_type TEXT        NOT NULL CHECK (source_type IN ('corpus', 'resume')),
  metadata    JSONB       NOT NULL DEFAULT '{}',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW index for fast ANN search
-- m=16: good recall/speed balance for < 100k vectors
-- ef_construction=64: solid build quality; ef_search tunable at query time
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
  ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- GIN full-text index for keyword/BM25 fallback
CREATE INDEX IF NOT EXISTS documents_fts
  ON documents
  USING gin(to_tsvector('english', content));

-- Source type index for filtered queries
CREATE INDEX IF NOT EXISTS documents_source_type_idx
  ON documents (source_type);

-- Source index for idempotent upserts (delete-by-source)
CREATE INDEX IF NOT EXISTS documents_source_idx
  ON documents (source);
