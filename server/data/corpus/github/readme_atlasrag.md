# GitHub README: AtlasRAG

Repository: https://github.com/vanshnarang13/AtlasRAG

# AtlasRAG

A production-grade **Multi-Modal Retrieval-Augmented Generation (RAG)** platform that lets users build AI-powered knowledge bases from documents and websites. Upload PDFs, crawl URLs, and ask natural language questions. The system retrieves relevant context and streams accurate, cited answers in real time.

## Architecture

<p align="center">
  <img src="https://ik.imagekit.io/5wegcvcxp/Resume-Multi-modal-rag/Local-Architecture.png" alt="Local Setup Architecture" width="100%">
</p>

### RAG Pipelines: Ingestion · Retrieval · Generation

<p align="center">
  <img src="https://ik.imagekit.io/5wegcvcxp/Resume-Multi-modal-rag/Indexing.png" alt="Ingestion Pipeline" width="100%">
</p>
<p align="center">
  <img src="https://ik.imagekit.io/5wegcvcxp/Resume-Multi-modal-rag/Retrieval.png" alt="Retrieval Pipeline" width="100%">
</p>
<p align="center">
  <img src="https://ik.imagekit.io/5wegcvcxp/Resume-Multi-modal-rag/Generation.png" alt="Generation Pipeline" width="100%">
</p>

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| **Backend** | FastAPI, Python, Uvicorn |
| **RAG / Agents** | LangChain, LangGraph, OpenAI GPT-4o / GPT-4 Turbo |
| **Embeddings** | OpenAI `text-embedding-3-large` (1536 dims) |
| **Vector DB** | Supabase (PostgreSQL + pgvector) |
| **Document Parsing** | Unstructured (PDF, DOCX, tables, images) |
| **Task Queue** | Celery + Redis |
| **Auth** | Clerk |
| **File Storage** | AWS S3 (presigned URL uploads) |
| **Web Scraping** | ScrapingBee |
| **Web Search** | Tavily |
| **Observability** | structlog (JSON), LangSmith tracing, RAGAS evaluation |
| **Containerization** | Docker + Docker Compose |

---

## Key Features

### Multi-Modal Document Processing
- Ingests PDFs, DOCX, and other file types alongside web URLs
- Extracts text, tables, and images using [Unstructured](https://github.com/Unstructured-IO/unstructured)
- Generates AI summaries for tables and images using GPT-4 Turbo before vectorizing, preserving fidelity for retrieval

### Flexible RAG Strategies (per-project, configurable)
| Strategy | Description |
|---|---|
| **Basic** | Pure vector similarity search |
| **Hybrid** | Vector + full-text keyword search fused with RRF (Reciprocal Rank Fusion) |
| **Multi-Query Vector** | Generates 5 query variants, searches each, fuses results with RRF |
| **Multi-Query Hybrid** | Query variants × hybrid search per variant, then RRF fusion |

### LangGraph Agent System
- **Simple RAG Agent**: enforces guardrails (toxicity, prompt injection, PII) before retrieval; maintains 10-message chat history
- **Supervisor Agent**: multi-tool orchestration with RAG + optional web search
- System prompts force tool use before generation; citations extracted and surfaced inline


### Async Document Ingestion Pipeline
Documents move through a tracked status machine:
```
uploading → pending → processing → partitioning → chunking → summarizing → vectorization → completed
```
Celery workers handle ingestion off the request path; clients poll for progress updates.

### Per-Project RAG Settings
Each project exposes a full configuration surface:
- RAG strategy, agent type, chunks per search, final context size
- Similarity threshold, number of multi-queries
- Reranking toggle + model (`reranker-english-v3.0`)
- Hybrid vector/keyword weights (default 0.7 / 0.3)

---

## Project Structure

```
AtlasRAG/
├── server/
│   ├── src/
│   │   ├── server.py           # FastAPI app entrypoint
│   │   ├── routes/             # 4 routers: users, projects, files, chats
│   │   ├── rag/
│   │   │   ├── ingestion/      # Partition → chunk → summarize → vectorize
│   │   │   └── retrieval/      # Basic, hybrid, multi-query strategies
│   │   ├── agents/
│   │   │   ├── simple_agent/   # LangGraph RAG agent with guardrails
│   │   │   └── supervisor_agent/
│   │   ├── services/           # Supabase, Clerk, S3, Celery, LLM clients
│   │   ├── models/             # Pydantic request/response schemas
│   │   ├── middleware/         # Auth + structured logging middleware
│   │   └── config/
│   ├── supabase/               # Local Supabase config + DB migrations
│   ├── evaluation/             # RAGAS evaluation scripts
│   ├── docker-compose.yml
│   └── Makefile
└── client/
    └── src/
        ├── app/                # Next.js App Router pages
        │   ├── projects/       # Project list + detail pages
        │   └── projects/[id]/chats/[chatId]/  # Chat interface
        ├── components/
        │   ├── chat/           # ChatInterface, MessageList, MessageItem, citations
        │   └── projects/       # ProjectsGrid, KnowledgeBaseSidebar, FileDetailsModal
        └── lib/                # API client, utilities
```

---

## API Reference

### Projects
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/projects/` | List user's projects |
| `POST` | `/projects/` | Create project |
| `GET` | `/projects/{id}` | Get project detail |
| `DELETE` | `/projects/{id}` | Delete project (cascades) |
| `GET` | `/projects/{id}/settings` | Get RAG configuration |
| `PUT` | `/projects/{id}/settings` | Update RAG configuration |

### Knowledge Base (Files & URLs)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/projects/{id}/files/` | List documents |
| `POST` | `/projects/{id}/files/upload-url` | Get presigned S3 upload URL |
| `POST` | `/projects/{id}/files/confirm` | Confirm upload → trigger ingestion |
| `POST` | `/projects/{id}/files/urls` | Add URL source |
| `DELETE` | `/projects/{id}/files/{fileId}` | Delete document |
| `GET` | `/projects/{id}/files/{fileId}/chunks` | Inspect chunks + embeddings |

### Chat
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chats/` | Create new chat |
| `GET` | `/chats/{id}` | Get chat with messages |
| `DELETE` | `/chats/{id}` | Delete chat |
| `POST` | `/projects/{id}/chats/{chatId}/messages` | Send message (JSON response) |
| `POST` | `/projects/{id}/chats/{chatId}/messages/stream` | Send message (SSE stream) |

---

## Data Model

<p align="center">
  <img src="docs/assets/database-erd.png" alt="AtlasRAG Database Schema ERD" width="100%">
</p>

<p align="center"><a href="https://app.eraser.io/workspace/3vKmgdruXhRmS9WxlsGN?diagram=-WOIPKRvViyjgT74km3o&layout=canvas">View/edit this ERD on Eraser</a></p>

## Setup

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Node.js](https://nodejs.org/) and npm
- Supabase CLI:

```bash
npm install -g supabase
```

---

### Step 1: Initialize Supabase

> **Important:** Supabase must be running before starting Docker containers.

**Start Supabase** (spins up Postgres, Auth, APIs, and provides local URLs/keys):

```bash
cd server
npx supabase start
```

**Apply migrations** (creates all tables fresh):

```bash
npx supabase db reset
```


**Check status** (note the `API URL` and `service_role key`, you'll need them):

```bash
npx supabase status
```

---

### Step 2: Configure Environment Variables

```bash
cd server
cp .env.sample .env
```

Open `.env` and fill in:


```bash
SUPABASE_API_URL= # Auto Configure in docker-compose file if running locally with containers
SUPABASE_SECRET_KEY=

CLERK_SECRET_KEY=
DOMAIN=


AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
S3_BUCKET_NAME=


REDIS_URL= # Auto Configure in docker-compose file if running locally with containers

OPENAI_API_KEY=

SCRAPINGBEE_API_KEY=

LANGSMITH_TRACING=
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=

TAVILY_API_KEY=

```

---

### Step 3: Start Docker Services

```bash
make start
```

This checks Supabase is running, then builds and starts the API server, Celery worker, and Redis.

**Other commands:**

```bash
make stop      # Stop all containers
make restart   # Restart all containers
make clean     # Remove containers, volumes, and images
```

---

### Step 4: Create a User Account

All API endpoints require authentication. You must provision a user before using the API.

1. Open [http://localhost:8000/docs](http://localhost:8000/docs)
2. Call `POST /api/user/create` with a [Clerk webhook payload](https://clerk.com/docs/guides/development/webhooks/overview#payload-structure)

---

### Step 5: Start the Frontend

```bash
cd client
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

### Logs & Monitoring

```bash
make logs-api     # API server logs
make logs-redis   # Redis logs
make logs-worker  # Celery worker logs
```