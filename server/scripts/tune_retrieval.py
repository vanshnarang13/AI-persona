"""
Retrieval hyperparameter grid search.

This script evaluates retrieval directly against the golden set. It does not
ask the LLM to answer; it measures whether the expected source appears in the
top results and how quickly retrieval runs.

Run from server/:
    PYTHONPATH=. python scripts/tune_retrieval.py
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

import asyncpg
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from src.rag.retrieval.vector import embed_query, vector_search
from src.rag.retrieval.lexical_bm25 import get_or_build_index, invalidate_index
from src.rag.retrieval.hybrid_rrf import rrf_fuse
from src.services.redis_client import get_redis_binary, close_redis

GOLDEN_PATH = Path(__file__).parent.parent / "evaluation" / "golden_set" / "questions.json"
RESULTS_DIR = Path(__file__).parent.parent / "evaluation" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Grid intentionally stays compact enough to run often after corpus updates.
GRID = {
    "top_k": [5, 8, 10],
    "candidate_multiplier": [2, 3, 4],
    "rrf_vector_w": [0.55, 0.65, 0.75, 0.85],
    "hnsw_ef_search": [64, 100, 200],
}

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _expected_sources(q: dict[str, Any]) -> list[str]:
    raw = q.get("expected_source")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [str(raw)]


def _source_hit(source: str, expected_sources: list[str]) -> bool:
    return any(expected in source for expected in expected_sources)


def _load_retrieval_cases() -> list[dict[str, Any]]:
    questions = json.loads(GOLDEN_PATH.read_text())
    cases = [q for q in questions if _expected_sources(q)]
    return cases


async def _delete_retrieval_cache(redis) -> int:
    deleted = 0
    async for key in redis.scan_iter(match="retrieve:*", count=500):
        await redis.delete(key)
        deleted += 1
    return deleted


async def evaluate_config(
    queries_with_embeddings: list[tuple[dict[str, Any], list[float]]],
    pool: asyncpg.Pool,
    bm25_index,
    top_k: int,
    candidate_multiplier: int,
    rrf_vector_w: float,
    hnsw_ef_search: int,
) -> dict[str, float]:
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    rrf_bm25_w = round(1.0 - rrf_vector_w, 2)
    candidate_k = top_k * candidate_multiplier

    for q, embedding in queries_with_embeddings:
        expected_sources = _expected_sources(q)

        t0 = time.perf_counter()
        vec_results = await vector_search(embedding, candidate_k, hnsw_ef_search, pool)
        bm25_results = bm25_index.search(q["question"], candidate_k)
        fused = rrf_fuse(vec_results, bm25_results, w_a=rrf_vector_w, w_b=rrf_bm25_w)
        latencies.append((time.perf_counter() - t0) * 1000)

        ranks = [
            idx
            for idx, chunk in enumerate(fused[: max(top_k, 5)], start=1)
            if _source_hit(chunk.source, expected_sources)
        ]
        if ranks:
            rank = ranks[0]
            reciprocal_ranks.append(1.0 / rank)
            if rank <= 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 5:
                hits_at_5 += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(queries_with_embeddings)
    return {
        "p_at_1": hits_at_1 / n,
        "p_at_3": hits_at_3 / n,
        "p_at_5": hits_at_5 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "avg_latency_ms": sum(latencies) / n,
    }


async def main() -> None:
    dsn = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
    redis = get_redis_binary()

    print("Invalidating BM25 and retrieval caches...")
    await invalidate_index(redis)
    deleted = await _delete_retrieval_cache(redis)
    print(f"Deleted {deleted} retrieval cache entries")

    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5, statement_cache_size=0)
    try:
        print("Building fresh BM25 index...")
        bm25_index = await get_or_build_index(pool, redis)

        cases = _load_retrieval_cases()
        print(f"Loaded {len(cases)} retrieval-labelled golden questions")

        print("Pre-computing query embeddings...")
        queries_with_embeddings: list[tuple[dict[str, Any], list[float]]] = []
        for q in cases:
            emb = await embed_query(q["question"])
            queries_with_embeddings.append((q, emb))

        total_configs = len(list(itertools.product(*GRID.values())))
        print(f"\nRunning grid search ({total_configs} configs)...\n")

        results: list[dict[str, Any]] = []
        for top_k, candidate_multiplier, rrf_vector_w, hnsw_ef_search in itertools.product(*GRID.values()):
            config = {
                "top_k": top_k,
                "candidate_multiplier": candidate_multiplier,
                "rrf_vector_w": rrf_vector_w,
                "rrf_bm25_w": round(1.0 - rrf_vector_w, 2),
                "hnsw_ef_search": hnsw_ef_search,
            }
            metrics = await evaluate_config(
                queries_with_embeddings,
                pool,
                bm25_index,
                top_k=top_k,
                candidate_multiplier=candidate_multiplier,
                rrf_vector_w=rrf_vector_w,
                hnsw_ef_search=hnsw_ef_search,
            )
            row = {**config, **metrics}
            results.append(row)
            print(
                "  "
                f"p@1={metrics['p_at_1']:.2f} "
                f"p@3={metrics['p_at_3']:.2f} "
                f"p@5={metrics['p_at_5']:.2f} "
                f"mrr={metrics['mrr']:.2f} "
                f"lat={metrics['avg_latency_ms']:.0f}ms "
                f"{config}"
            )

        results.sort(key=lambda r: (-r["p_at_3"], -r["mrr"], -r["p_at_1"], r["avg_latency_ms"]))
        best = results[0]

        out_path = RESULTS_DIR / f"retrieval_tuning_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps({"best": best, "results": results}, indent=2))

        print(f"\n{'=' * 60}")
        print(f"{GREEN}Best retrieval config{RESET}")
        print(f"  p@1      : {best['p_at_1']:.1%}")
        print(f"  p@3      : {best['p_at_3']:.1%}")
        print(f"  p@5      : {best['p_at_5']:.1%}")
        print(f"  MRR      : {best['mrr']:.3f}")
        print(f"  latency  : {best['avg_latency_ms']:.0f}ms avg, excluding embed")
        print("  params   :")
        print(f"    TOP_K_CHAT           = {best['top_k']}")
        print(f"    CANDIDATE_MULTIPLIER = {best['candidate_multiplier']}")
        print(f"    VECTOR_WEIGHT        = {best['rrf_vector_w']}")
        print(f"    BM25_WEIGHT          = {best['rrf_bm25_w']}")
        print(f"    HNSW_EF_SEARCH       = {best['hnsw_ef_search']}")
        print(f"\n  Saved full tuning results: {out_path}")
        if best["p_at_3"] < 0.9:
            print(f"{YELLOW}  p@3 is below 90%; inspect misses before applying blindly.{RESET}")
        print(f"{'=' * 60}")
    finally:
        await pool.close()
        await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
