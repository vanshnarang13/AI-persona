# GitHub README: TradeSmith

Repository: https://github.com/vanshnarang13/TradeSmith

# TradeSmith

An AI-powered autonomous trading platform where four agents modeled after Warren Buffett, George Soros, Ray Dalio, and Cathie Wood research the market and execute trades independently using real-time stock data.

Each agent runs its own investment strategy, manages a simulated $10,000 portfolio, and logs every decision. A live Gradio dashboard lets you watch portfolio values, holdings, and trade history update in real time.

---

## Getting Started

### Prerequisites

- Python **3.12+**
- API keys for the services listed in `.env.sample` (Polygon and at least one LLM provider are the minimum)

### 1. Clone the repo

```bash
git clone https://github.com/vanshnarang/TradeSmith.git
cd TradeSmith
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.sample .env
```

Open `.env` and fill in your API keys. At minimum you need:

| Variable | Where to get it |
|---|---|
| `POLYGON_API_KEY` | [polygon.io](https://polygon.io) — free tier works |
| `BRAVE_API_KEY` | [api.search.brave.com](https://api.search.brave.com) |
| `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai) |

See `.env.sample` for all optional variables (multi-model mode, push notifications, etc.).

### 5. Initialise trader accounts

This creates the SQLite database and seeds each trader with a $10,000 starting balance.

```bash
python -m src.scripts.reset
```

### 6. Run the trading floor

Open two terminals (or use a process manager like `tmux`):

**Terminal 1 — trading scheduler** (agents run every N minutes):
```bash
python -m src.scripts.trading_floor
```

**Terminal 2 — monitoring dashboard** (Gradio UI, opens in your browser):
```bash
python -m src.app.app
```

The dashboard auto-refreshes every 120 seconds and shows live P&L, holdings, and agent activity logs.

---

## Traders

| Agent | Strategy | Based on |
|---|---|---|
| **Warren** | Value investing — undervalued stocks, long-term holds | Warren Buffett |
| **George** | Macro trading — contrarian bets on large market moves | George Soros |
| **Ray** | Risk parity — diversified, systematic allocation | Ray Dalio |
| **Cathie** | Disruptive innovation — growth stocks and crypto ETFs | Cathie Wood |

---

## Architecture

```
TradeSmith/
├── .env.sample             # Environment variable template
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
└── src/
    ├── agents/
    │   └── traders.py      # Agent logic (researcher + trader pair per strategy)
    ├── app/
    │   └── app.py          # Gradio dashboard
    ├── clients/
    │   └── accounts_client.py  # MCP client for account operations
    ├── config/
    │   └── mcp_params.py   # MCP server configuration
    ├── core/
    │   └── database.py     # SQLite layer (accounts, logs, market cache)
    ├── models/
    │   ├── accounts.py     # Account management and portfolio calculations
    │   └── market.py       # Market data via Polygon API
    ├── scripts/
    │   ├── reset.py        # Seed/reset trader accounts
    │   └── trading_floor.py  # Scheduler — runs agents on a configurable interval
    ├── servers/
    │   ├── accounts_server.py  # MCP server — buy/sell, balance checks
    │   ├── market_server.py    # MCP server — live stock quotes
    │   └── push_server.py      # MCP server — Pushover notifications
    └── utils/
        ├── templates.py    # Prompt templates for agents
        ├── tracers.py      # Logging and tracing
        └── util.py         # Dashboard helpers (CSS, colours)
```

**How it works:** each trader is a two-tier agent pair. A *researcher* agent searches the web and pulls market data; a *trader* agent reads that research and decides what to buy, sell, or hold. All decisions are executed through MCP servers so the trading logic stays cleanly separated from the LLM layer.

---

## Features

- Four autonomous AI agents with distinct, realistic investment strategies
- Real-time (or 15-min delayed) market data via Polygon.io
- MCP-based tool architecture — agents call typed tools, not raw strings
- Multi-model support: run each trader on a different LLM (OpenAI, DeepSeek, Gemini, Grok)
- SQLite persistence — accounts, transactions, and agent traces survive restarts
- Gradio dashboard with live portfolio charts, holdings table, and activity log
- Optional Pushover push notifications on every trade