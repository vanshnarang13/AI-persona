# Project: TradeSmith

## Overview

TradeSmith is an autonomous multi-agent trading platform where four AI agents, each modeled after a legendary investor, independently research the market and execute real trades against simulated portfolios. Each agent starts with $10,000, applies its own investment philosophy, and runs on a schedule without any human in the loop. A Gradio dashboard displays live portfolio values, current holdings, and trade history.

The deeper goal of the project was to explore whether distinct LLM-powered investment personas behave meaningfully differently when given access to the same market data and tools. The answer is yes: Warren holds positions for longer, Cathie concentrates in growth and innovation stocks, George takes contrarian bets based on macro themes, and Ray tries to maintain diversification.

GitHub: https://github.com/vanshnarang13/TradeSmith

## Tech Stack

Python 3.12, OpenAI Agents SDK, FastMCP, MCP (Model Context Protocol), Polygon.io API (market data), Brave Search API (web research), SQLite (portfolio and trade log storage), libsql (SQLite-compatible in-process memory store for per-agent memory), Gradio (live dashboard), asyncio (concurrency), Pushover (push notifications on every trade), OpenRouter (multi-model routing), DeepSeek V3, Gemini 2.5 Flash, Grok 3 Mini, uvx and uv (dependency management).

## The Four Agents

Warren models Warren Buffett's value investing philosophy. He looks for undervalued companies with durable competitive advantages and holds positions long term. He is skeptical of high-growth speculative positions.

George models George Soros's macro trading approach. He takes contrarian bets on large market movements, often positioning against consensus expectations when he sees a fundamental misalignment.

Ray models Ray Dalio's risk parity principles. He favors diversification, systematic allocation, and tends to avoid large concentrated positions in any single asset.

Cathie models Cathie Wood's disruptive innovation thesis. She concentrates in high-growth technology and innovation stocks, including ETFs with cryptocurrency exposure, and has a longer time horizon for disruptive themes.

## Architecture

```
Trading Floor Scheduler (asyncio event loop)
    |
    4 Trader agents running in parallel on a schedule
         |
         Each Trader consists of:
              |--- Researcher sub-agent
              |         Uses MCP tools: Brave Search web fetch, libsql memory read/write
              |         Produces a research report injected into the Trader's context
              |
              |--- Trader agent
                        Uses MCP tools: get_balance, get_holdings, buy_shares, sell_shares, change_strategy
                        Uses MCP tools: lookup_share_price (Polygon.io), send_notification (Pushover)

MCP Servers (stdio transport, launched per run):
    |--- accounts_server.py  : get_balance, get_holdings, buy_shares, sell_shares, change_strategy
    |--- market_server.py    : lookup_share_price via Polygon.io
    |--- push_server.py      : Pushover push notification per trade

Gradio Dashboard:
    |--- Auto-refresh every 120 seconds
    |--- Shows P&L per agent, holdings table, decision log with reasoning
```

## Key Design Decisions

### Two-Tier Agent Architecture

Each agent is actually two agents: a Researcher and a Trader. The Researcher runs first. It searches the web using Brave Search, reads the agent's own memory store for past investment theses and notes, and produces a research report. That report is injected as a tool call result into the Trader agent's context. The Trader then reads the research and decides what to buy, sell, or hold.

This separation was a deliberate design choice for two reasons. First, it keeps the web content out of the Trader's direct context. Raw web-scraped content can contain adversarial text. If you inject news articles directly into the Trader's reasoning, you have a prompt injection surface. Keeping the Researcher as a separate agent that mediates the web data gives you one more layer of separation. Second, it forces the Trader to operate on synthesized intelligence (the research report) rather than raw data, which produces cleaner reasoning.

### MCP for Tool Separation

All trading operations, meaning balance checks, buy and sell execution, market data lookups, and push notifications, are exposed as typed MCP tools through FastMCP stdio servers. The LLM never calls raw Python functions. It calls named tools with structured arguments.

This decouples the trading logic entirely from the LLM layer. If you want to swap the underlying brokerage from simulated to real, you rewrite the MCP server and nothing else changes. If you want to add a new capability, you add a new tool. The LLM orchestration code does not need to know how any of the tools are implemented.

### Per-Agent Persistent Memory

Each Researcher agent has its own libsql memory store at `memory/{agent_name}.db`. After each run, the agent can write notes to its memory about what it researched, what it concluded, and what it is watching. On the next run, it reads those notes before starting new research. This allows agents to build investment theses across cycles rather than starting fresh each time.

Warren, for example, can note "monitoring $AAPL for entry below $170" and recall that context in the next research cycle.

### Alternating Trade and Rebalance Cycles

Each agent alternates between two modes. In trade mode, it runs the full research pipeline and then executes new positions based on its findings. In rebalance mode, it reviews its current holdings and makes smaller adjustments. A `do_trade` boolean flips each cycle. This prevents the agents from churning excessively while still allowing them to respond to new information regularly.

### Multi-Model Support

Each of the four agents can run on a different LLM. The default configuration uses GPT-4.1 Mini for all four, but the system supports running Warren on GPT-4.1 Mini, George on DeepSeek V3, Ray on Gemini 2.5 Flash, and Cathie on Grok 3 Mini simultaneously, controlled by a `USE_MANY_MODELS` environment variable. This lets you observe whether different LLMs exhibit systematically different trading behaviors even when given the same persona instructions.

## MCP Server Lifecycle Management

Launching and shutting down MCP stdio servers is not trivial. Each run requires starting three MCP servers per agent (accounts, market, and push) before the agent can use any tools, and shutting them down cleanly after. With four agents running in parallel, that is twelve server processes managed simultaneously.

This was handled using Python's `AsyncExitStack`. Each MCP client connection is entered as an async context manager, and the stack handles cleanup in the correct order even if there are failures. Without this, server processes would leak between runs.

## Results and Findings

The four agents do exhibit distinct behaviors that match their investor personas. Warren consistently holds positions longer and has fewer total trades. Cathie consistently takes larger positions in technology and innovation ETFs. George makes more contrarian bets. Ray maintains better portfolio diversification.

All four agents correctly respect portfolio constraints enforced by the MCP tool layer. When an agent tries to buy more shares than its balance allows, the accounts server returns an error and the agent adjusts. The LLM never handles financial math directly, which is intentional. The tool layer does the arithmetic.

Every trade decision is logged with the agent's full reasoning, which makes post-hoc analysis of decision quality straightforward.

## Challenges and How They Were Solved

Managing concurrent SQLite access from four parallel agents was the first real challenge. SQLite has limited support for concurrent writers. The solution was careful connection management with explicit transaction boundaries and serialized writes to shared tables.

The MCP server lifecycle problem (managing twelve stdio processes per run cycle) was solved with AsyncExitStack as described above.

Prompt injection from news content was addressed architecturally by the two-tier Researcher and Trader separation, not by trying to sanitize web content.

## What I Would Do Differently

I would replace SQLite with PostgreSQL for the portfolio database. SQLite's concurrency limitations are a real constraint as the number of agents scales. Postgres handles concurrent writes correctly without needing explicit serialization.

I would add a fifth agent as a portfolio risk manager. This agent would have no trading capability but would review the aggregate positions across all four traders and veto any trade that would cause the overall portfolio to exceed position-size limits or correlation thresholds. Right now the four traders self-regulate but they have no awareness of each other's positions. A dedicated risk layer would be safer in a real deployment.

I would also implement proper backtesting infrastructure before running the agents forward. The current system runs in real time (against delayed quotes on the free Polygon tier). Being able to replay the system against historical data would make it much easier to validate whether the persona differences are real and consistent or just noise.
