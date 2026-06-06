# Project: AtlasRAG

## Overview

AtlasRAG is a production-grade multimodal RAG (Retrieval-Augmented Generation) platform that lets users build private AI knowledge bases from their own documents and websites. A user uploads PDFs, DOCX files, PowerPoints, or provides URLs, and the system ingests those documents asynchronously in the background, vectorizes the content, and makes it queryable through a chat interface. When the user asks a question, the system retrieves the most relevant chunks, passes them to an LLM, and streams back a cited answer.

This was built as a real product with auth, per-project settings, background job processing, a proper database schema, and a deployed frontend. It is not a Jupyter notebook or a demo with hardcoded documents.

GitHub: https://github.com/vanshnarang13/AtlasRAG

## Tech Stack

Python, FastAPI, Uvicorn, LangChain, LangGraph, OpenAI GPT-4o and GPT-4o-mini (generation), OpenAI GPT-4 Turbo (multimodal summarization), OpenAI text-embedding-3-large at 1536 dimensions (embeddings), Supabase (PostgreSQL plus pgvector), Unstructured library (document parsing), Celery (async task queue), Redis (Celery broker and cache), Clerk (authentication and JWT validation), AWS S3 (file storage with presigned URLs), ScrapingBee (web crawling), Tavily (web search), DuckDuckGo (fallback search), LangSmith (tracing), RAGAS (evaluation), structlog (structured JSON logging), Docker and Docker Compose, Next.js 16, React 19, TypeScript, Tailwind CSS 4.

## Architecture

The system has two main data paths: the ingestion pipeline and the retrieval pipeline.

The ingestion pipeline is asynchronous. When a user uploads a file, the API immediately returns a document ID and sets the status to "uploading." The actual processing happens in a Celery worker. The worker moves the document through nine states: uploading, pending, partitioning, chunking, summarizing, vectorization, completed, and two failure states. The frontend polls for status changes and shows a progress indicator. This design means the API is never blocked by expensive processing work.

The retrieval pipeline is synchronous and per-request. When a user sends a chat message, the API runs the selected RAG strategy, feeds the retrieved chunks and conversation history to a LangGraph agent, and streams the response back via Server-Sent Events.

```
Client (Next.js 16)
    |
    REST API (FastAPI)
         |--- Auth middleware (Clerk JWT validation)
         |--- Routes: /users, /projects, /projects/{id}/files, /chats
         |
         |--- Ingestion pipeline (Celery async workers)
         |         uploading > pending > partitioning > chunking > summarizing > vectorization > completed
         |
         |--- Retrieval + Agent pipeline (sync, per-request)
                   |--- Strategy selection (basic, hybrid, multi-query-vector, multi-query-hybrid)
                   |--- Agent selection (simple_agent or supervisor_agent)
                   |--- SSE streaming response
```

## Ingestion Pipeline in Detail

Step 1 (Partition): The file is downloaded from S3, or the URL is crawled using ScrapingBee. The Unstructured library extracts elements from the document. For PDFs and DOCX files this produces a mix of text blocks, table objects, and image objects. For websites it extracts the main body text.

Step 2 (Chunk): Chunks are produced using Unstructured's `chunk_by_title` strategy with a maximum of 3000 characters, a soft limit of 2400 characters, and a merge threshold of 500 characters (very short chunks are merged with adjacent ones). This produces semantically coherent chunks that respect document structure.

Step 3 (Summarize): This is the interesting step. For chunks that contain tables or images, a GPT-4 Turbo call generates an AI description of the content. The model is shown the table or image and asked to describe it in a way that would be useful for retrieval. The AI-generated description is what gets vectorized. For pure text chunks, no summarization is needed and they pass through directly. This solves a real problem: you cannot embed a raw table in a useful way. Raw table HTML or CSV text embeds very poorly because it has no semantic density. A natural language description of what the table contains embeds much better.

Step 4 (Vectorize): The text content (original text or AI summary) is embedded in batches of 10 using `text-embedding-3-large` at 1536 dimensions with exponential backoff retry on API failures. Embeddings and their original content are stored in Supabase's pgvector table. The original content is preserved separately from the AI summary so the UI can display the source faithfully.

## Retrieval Strategies

Every project has its own configured strategy. This is a per-project setting stored in the database, not a global config.

Basic: pure cosine similarity search using pgvector. Fast and simple but misses keyword-heavy queries.

Hybrid: vector similarity combined with PostgreSQL full-text search. Results from both are fused using Reciprocal Rank Fusion with configurable weights (default 0.7 for vector, 0.3 for keyword). This handles both semantic queries ("explain the revenue model") and exact term queries ("what is the EBITDA margin in Q3").

Multi-query vector: the user's question is rewritten into N variations by an LLM (e.g. "what is the budget?" might become "total allocated funds", "spending limit", "financial cap"). Vector search runs for each variation separately and results are fused with RRF. This helps when the user's phrasing does not match the document's phrasing.

Multi-query hybrid: same as multi-query vector but using the hybrid retrieval per variation. The most thorough strategy and the most expensive.

## Agent System

Simple agent: a LangGraph StateGraph with three nodes. The guardrail node runs first and checks the user's message for toxicity, prompt injection, and PII using GPT-4o-mini with structured output via a Pydantic schema called InputGuardrailCheck. If the guardrail passes, the agent node runs. The agent is forced to call the `rag_search` tool before generating any answer. Citations are accumulated in a custom state field across tool calls using a LangGraph reducer function.

Supervisor agent: a more complex graph where a supervisor LLM coordinates between two sub-agents. The RAG agent handles questions about the project documents. The Web Search agent uses Tavily (or DuckDuckGo as fallback) for external queries. The routing rule is: always use the RAG tool first, only use web search if the user explicitly asks for external information. Citations propagate up through LangGraph Command updates from sub-agents to the supervisor's state.

## Citation Tracking

This was a non-trivial engineering problem. LangGraph's default state update behavior replaces the previous value. So if you have a `citations` field in your state and the RAG tool adds two citations, then the agent calls the tool again, the second call would overwrite the first batch. The fix is a custom reducer: `lambda x, y: x + y` on the citations field. This tells LangGraph to concatenate new citations with existing ones rather than replacing them.

## Evaluation

There is a RAGAS evaluation pipeline in the `evaluation/` directory with a golden dataset of question-answer pairs. The metrics computed are faithfulness (does the answer only use information from the retrieved context?), answer relevancy (does the answer actually address the question?), context precision (how relevant are the retrieved chunks?), and context recall (does the retrieved context contain the answer?).

Key findings: hybrid and multi-query strategies consistently outperform basic vector search on recall for keyword-heavy queries. AI-summarized table chunks dramatically improve retrieval quality for documents with financial tables. The guardrail node adds 200 to 400 milliseconds of latency per request.

## Challenges and How They Were Solved

Celery workers in Docker needed to share database connections with the API container. This required careful environment variable handling and understanding the order in which the Supabase client initializes.

Multimodal chunk quality was a real problem early on. Embedding raw base64 image data or table HTML produces garbage retrieval results. The AI-summarize-before-vectorize approach solved this completely. The insight is that you want to embed a description of what the chunk means, not the raw bytes.

Citation accumulation across multiple tool calls in LangGraph required the custom reducer pattern described above. LangGraph's default state update semantics are not obvious and this was a source of bugs before the reducer approach was clear.

## What I Would Do Differently

I would add SSE-based ingestion status streaming instead of client polling. Right now the frontend polls a status endpoint every few seconds. That works but it is not elegant and it generates unnecessary database reads. With SSE the server can push status updates as they happen.

The cross-encoder reranker is already in the project settings schema (there is a field for it) but it is not wired up in the retrieval pipeline. Adding Cohere's reranker-english-v3.0 as a final reranking step after retrieval would improve precision by roughly 10 to 15 percent based on benchmarks.

I would also move from per-request agent instantiation to persistent agent sessions with conversation memory stored in the database. Right now conversation history is injected into the system prompt, which works but is not the right architecture for long sessions.
