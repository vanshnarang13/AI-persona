"""
Fetch commit history and READMEs from Vansh's public GitHub repos and write
them as corpus markdown files under data/corpus/github/.

Then re-ingest the corpus so the vector store is up to date.

Usage:
    cd server
    PYTHONPATH=. python ingest/ingest_github.py

Optional env vars:
    GITHUB_TOKEN  — personal access token (avoids 60 req/hr unauthenticated limit)

What this produces per repo:
    data/corpus/github/commits_<repo>.md   — last 30 commits with message + files changed
    data/corpus/github/readme_<repo>.md    — raw README content from the repo

Why pre-ingest instead of a runtime GitHub tool:
    Runtime tools add 500ms–2s latency per query and fail on API rate limits.
    These repos are complete projects whose history rarely changes.
    Pre-ingesting gives zero-latency grounded answers to commit history questions,
    which is specifically tested as an adversarial eval case in the assignment.
    Trade-off: data is frozen at ingest time. A scheduled re-ingest or GitHub
    push webhook would keep it fresh for a long-running deployment.
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
OWNER = "vanshnarang13"
COMMITS_PER_REPO = 30

REPOS = [
    {"repo": "AtlasRAG",                          "label": "AtlasRAG"},
    {"repo": "TradeSmith",                        "label": "TradeSmith"},
    {"repo": "Drone-Audio-Classification",        "label": "Drone Audio Classification"},
    {"repo": "Multimodal-Property-Price-Prediction", "label": "Multimodal Property Price Prediction"},
    {"repo": "Stock-Sentiment-Analysis",          "label": "Stock Sentiment Analysis"},
    {"repo": "sar-to-eo",                         "label": "SAR to EO Image Translation"},
]

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "corpus" / "github"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


async def fetch_commits(client: httpx.AsyncClient, repo: str) -> list[dict]:
    url = f"https://api.github.com/repos/{OWNER}/{repo}/commits"
    resp = await client.get(url, headers=_headers(), params={"per_page": COMMITS_PER_REPO})
    if resp.status_code == 404:
        print(f"  {YELLOW}repo not found: {repo}{RESET}")
        return []
    resp.raise_for_status()
    return resp.json()


async def fetch_commit_detail(client: httpx.AsyncClient, repo: str, sha: str) -> dict:
    url = f"https://api.github.com/repos/{OWNER}/{repo}/commits/{sha}"
    resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def fetch_readme(client: httpx.AsyncClient, repo: str) -> str:
    url = f"https://api.github.com/repos/{OWNER}/{repo}/readme"
    resp = await client.get(url, headers=_headers())
    if resp.status_code == 404:
        return ""
    resp.raise_for_status()
    data = resp.json()
    content = data.get("content", "")
    if content:
        return base64.b64decode(content).decode("utf-8", errors="replace")
    return ""


def _format_commits_md(label: str, repo: str, commits: list[dict]) -> str:
    lines = [
        f"# GitHub Commit History: {label}",
        "",
        f"Repository: https://github.com/{OWNER}/{repo}",
        f"Last {len(commits)} commits (most recent first)",
        "",
    ]

    for c in commits:
        commit = c.get("commit", {})
        sha = c.get("sha", "")[:8]
        msg = commit.get("message", "").strip()
        author = commit.get("author", {})
        date_str = author.get("date", "")
        name = author.get("name", "")

        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            date_fmt = dt.strftime("%d %b %Y")
        except Exception:
            date_fmt = date_str[:10]

        lines.append(f"## Commit {sha} — {date_fmt}")
        lines.append(f"Author: {name}")
        lines.append("")

        # Use only first line of commit message as heading, rest as body
        msg_lines = msg.split("\n")
        lines.append(f"Message: {msg_lines[0]}")
        if len(msg_lines) > 1:
            body = "\n".join(l for l in msg_lines[1:] if l.strip())
            if body:
                lines.append("")
                lines.append(body)

        # Files changed (from detail fetch, may be empty if not fetched)
        files = c.get("_files", [])
        if files:
            lines.append("")
            lines.append(f"Files changed ({len(files)}):")
            for f in files[:8]:
                fname = f.get("filename", "")
                status = f.get("status", "")
                additions = f.get("additions", 0)
                deletions = f.get("deletions", 0)
                lines.append(f"  {status:10s} {fname}  (+{additions} -{deletions})")
            if len(files) > 8:
                lines.append(f"  ... and {len(files) - 8} more files")

        lines.append("")

    return "\n".join(lines)


def _format_readme_md(label: str, repo: str, readme: str) -> str:
    return (
        f"# GitHub README: {label}\n\n"
        f"Repository: https://github.com/{OWNER}/{repo}\n\n"
        f"{readme}"
    )


async def process_repo(client: httpx.AsyncClient, entry: dict) -> tuple[int, int]:
    repo = entry["repo"]
    label = entry["label"]
    print(f"\n  Fetching {label} ({repo})...")

    commits_raw = await fetch_commits(client, repo)
    if not commits_raw:
        return 0, 0

    # Fetch detail for first 10 commits to get file lists (avoid rate limit on all 30)
    for i, c in enumerate(commits_raw[:10]):
        try:
            detail = await fetch_commit_detail(client, repo, c["sha"])
            c["_files"] = detail.get("files", [])
        except Exception:
            c["_files"] = []

    readme_text = await fetch_readme(client, repo)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    commits_md = _format_commits_md(label, repo, commits_raw)
    commits_path = OUTPUT_DIR / f"commits_{repo.lower().replace('-', '_')}.md"
    commits_path.write_text(commits_md, encoding="utf-8")
    print(f"  {GREEN}wrote{RESET} {commits_path.name}  ({len(commits_raw)} commits)")

    readme_written = 0
    if readme_text.strip():
        readme_md = _format_readme_md(label, repo, readme_text)
        readme_path = OUTPUT_DIR / f"readme_{repo.lower().replace('-', '_')}.md"
        readme_path.write_text(readme_md, encoding="utf-8")
        print(f"  {GREEN}wrote{RESET} {readme_path.name}  ({len(readme_text)} chars)")
        readme_written = 1

    return len(commits_raw), readme_written


async def main() -> None:
    print(f"\nGitHub ingestion — owner: {OWNER}")
    print(f"Token: {'set' if GITHUB_TOKEN else 'not set (60 req/hr limit applies)'}")
    print(f"Output: {OUTPUT_DIR}")

    total_commits = 0
    total_readmes = 0

    async with httpx.AsyncClient(timeout=20) as client:
        for entry in REPOS:
            try:
                commits, readmes = await process_repo(client, entry)
                total_commits += commits
                total_readmes += readmes
            except httpx.HTTPStatusError as e:
                print(f"  {RED}HTTP error for {entry['repo']}: {e.response.status_code}{RESET}")
            except Exception as e:
                print(f"  {RED}Error for {entry['repo']}: {e}{RESET}")

    print(f"\n{'='*50}")
    print(f"Fetched {total_commits} commits and {total_readmes} READMEs")
    print(f"Files written to: {OUTPUT_DIR}")
    print(f"{'='*50}")

    # Run corpus ingestion to index the new files
    print("\nRunning corpus ingestion...")
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.services.supabase_client import get_pool, close_pool
    from src.services.redis_client import get_redis_binary, close_redis
    from src.rag.ingestion import run_ingestion
    from src.rag.retrieval import invalidate_index

    pool = await get_pool()
    try:
        stats = await run_ingestion(OUTPUT_DIR, pool)
        redis = get_redis_binary()
        await invalidate_index(redis)
        deleted = 0
        async for key in redis.scan_iter(match="retrieve:*", count=500):
            await redis.delete(key)
            deleted += 1
        print(f"Ingested: {stats.get('stored', 0)} chunks from {stats.get('files', 0)} files")
        print(f"Cleared {deleted} retrieval cache entries")
    finally:
        await close_pool()
        await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
