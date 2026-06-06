# Skills — Vansh Narang

## What I Am Strongest At

Production agentic systems using LangGraph and the OpenAI Agents SDK, RAG pipeline design from ingestion through retrieval to generation, async Python backends with FastAPI and Celery, and deep learning model implementation starting from research papers. These are the areas where I have the most depth and the most production experience.

## Programming Languages

Python is my primary language. I use it for everything: ML models, backend APIs, data pipelines, and agentic systems. I also write JavaScript and TypeScript for frontend work (Next.js). SQL is something I write regularly for database queries, schema design, and pgvector operations.

## AI and Machine Learning

PyTorch: I use PyTorch for all deep learning work. I have implemented encoder-decoder architectures, GANs (SteganoGAN), attention-based models (CAISFormer), multi-task CNNs (drone audio classification), and multimodal fusion networks (ResNet plus tabular features for property price prediction). I understand training dynamics, loss function design, and debugging convergence failures.

TensorFlow and Keras: used for the drone audio classification project with the Keras functional API for multi-output model definition.

Scikit-Learn: standard ML toolkit for ensemble models (XGBoost, LightGBM, Ridge, Random Forest, MLP), feature engineering pipelines, and evaluation metrics.

Hugging Face Transformers: FLAN-T5 for text condensation (Amazon ML challenge)

XGBoost and LightGBM: used extensively for tabular ML, including ensembling with neural networks.


## LLM and Agentic Systems

LangGraph: I have built multiple LangGraph applications. AtlasRAG uses a StateGraph with guardrail nodes, agent nodes, and custom state reducers for citation accumulation. This AI persona system uses a StateGraph-based booking agent with InjectedState tools and AsyncPostgresSaver for Postgres-backed checkpointing. I understand how to design graph topology, handle state updates, and debug recursion limit issues.

LangChain: used within AtlasRAG for tool definitions, retrieval chain components, and LLM client abstractions.

OpenAI Agents SDK: TradeSmith is built on the OpenAI Agents SDK with a two-tier Researcher and Trader agent architecture. I understand how to compose agents, inject tool results between agents, and manage agent lifecycles.

CrewAI , Autogen

MCP (Model Context Protocol) and FastMCP: I have built MCP servers from scratch for TradeSmith. The accounts server, market data server, and push notification server are all custom FastMCP stdio servers. I understand the stdio transport, the tool schema definition, and how to manage server lifecycle with AsyncExitStack.

OpenAI API: generation (GPT-4o, GPT-4o-mini, GPT-4 Turbo), embeddings (text-embedding-3-small and text-embedding-3-large), and structured output with Pydantic schemas.

Vapi: voice AI platform. I built the voice integration for this AI persona system. I understand how Vapi sends tool calls (arguments as JSON objects, not strings), how to configure custom server tools with timeouts, how the Deepgram STT and ElevenLabs TTS are wired together, and how to debug call logs via the Vapi API.

Multi-model routing: TradeSmith routes across GPT-4.1 Mini, DeepSeek V3, Gemini 2.5 Flash, and Grok 3 Mini via OpenRouter.

RAG (Retrieval-Augmented Generation): I have built RAG systems from scratch twice, with different architectures. AtlasRAG uses four configurable retrieval strategies with vector search, hybrid search, and multi-query expansion. This persona system uses hybrid vector plus BM25 retrieval fused with RRF and cross-encoder reranking.

Hybrid retrieval: combining dense vector search (pgvector) with sparse keyword search (BM25 or PostgreSQL full-text search) and fusing results with Reciprocal Rank Fusion.

RRF (Reciprocal Rank Fusion): the standard approach for combining ranked lists from multiple retrieval sources. I have implemented and tuned RRF weights in production.

pgvector and HNSW: I set up pgvector with HNSW indexing (m=16, ef_construction=64) in Supabase for both AtlasRAG and this persona system. I understand the trade-off between ef_search and query latency.

Cross-encoder reranking: used in the chat path of this persona system to rerank fused retrieval results before passing them to the LLM.

Guardrails: structured output-based input validation (AtlasRAG uses GPT-4o-mini with Pydantic schema for toxicity, prompt injection, and PII detection), and context grounding checks (this persona system checks retrieval confidence before answering).

## Development and Infrastructure

FastAPI and Uvicorn: I build production FastAPI applications with SSE streaming, middleware for rate limiting and structured logging, dependency injection, and Pydantic request and response models.

Celery and Redis: async task queues for background document processing in AtlasRAG. Redis as both the Celery broker and a retrieval cache (binary client with pickle for BM25 and cache).

Next.js and React: I build frontends for my projects. The AtlasRAG frontend is a full Next.js application with TypeScript and Tailwind. This persona chat UI is also Next.js.

Docker and Docker Compose: I containerize all my applications for deployment. Both AtlasRAG and this persona system have Dockerfiles and compose configs.

PostgreSQL and Supabase: I write SQL for schema design, queries, and pgvector operations. Supabase is my hosted Postgres provider.

AWS S3: presigned URL-based file uploads in AtlasRAG.

Redis: binary client mode (decode_responses=False) for pickle-based BM25 index caching and retrieval caching. The binary client is essential because the BM25 index is serialized with pickle, not as a UTF-8 string.

asyncio: async Python throughout all backend systems. httpx.AsyncClient for all external HTTP calls.

Clerk: JWT-based authentication in AtlasRAG.

Gradio: live dashboard in TradeSmith.

SQLite and libsql: audit logs and per-agent memory in TradeSmith.

## Embeddings and Vector Search

OpenAI text-embedding-3-small (1536 dimensions): used for retrieval in this persona system and as the default embedding model.

OpenAI text-embedding-3-large (1536 dimensions): used in AtlasRAG for higher-quality embeddings on document content.

pgvector: PostgreSQL extension for storing and querying dense vectors. I have set up HNSW indexes and GIN full-text indexes in pgvector databases.

BM25: implemented rank_bm25.BM25Okapi for sparse keyword retrieval. The index is built at server startup from all document contents and cached in Redis as a pickled object.

## Observability and Evaluation

LangSmith: tracing for LangGraph and LangChain applications in AtlasRAG.

RAGAS: RAG evaluation framework. Metrics I have used: faithfulness, answer relevancy, context precision, and context recall. I have built golden evaluation datasets and run automated evaluation pipelines.

structlog: structured JSON logging in all production FastAPI applications. I log latency at each stage of the retrieval pipeline (embed_ms, vector_ms, bm25_ms, rrf_ms, rerank_ms, total_ms) for performance monitoring.

SQLite audit logs: per-decision logging in TradeSmith with full agent reasoning stored per trade.

Custom eval scripts: I wrote eval_vapi.py for end-to-end Vapi tool evaluation in this persona system, which runs retrieval quality checks, latency benchmarks, and call log analysis.

## Tools and Platforms

Git and GitHub, Jupyter Notebooks, uv and uvx (Python package management), Polygon.io API (market data), Brave Search API, ScrapingBee (web crawling), Tavily (web search), Pushover (push notifications), Vapi (voice AI), Cal.com (calendar booking), Render (deployment), Vercel, Upstash (managed Redis), ngrok (local tunnel for Vapi testing), MLflow (experiment tracking in property price prediction), Optuna (hyperparameter optimization).
