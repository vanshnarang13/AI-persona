# Fit for Role — Vansh Narang × Scaler AI Engineer Intern

## The Role

Scaler is building AI-native EdTech, specifically always-on autonomous learner agents. The AI Engineer Intern role requires production-quality Python, async systems, agentic pipelines, RAG, LLM orchestration, and the ability to ship end-to-end AI features fast.

## Why Vansh Fits

### Direct Product Overlap

Scaler is building autonomous learner agents. Vansh has built systems that are architecturally identical.

AtlasRAG is a production RAG platform with LangGraph agents, async Celery ingestion pipelines, hybrid retrieval combining vector search and BM25 keyword search with RRF fusion, and per-project configurable retrieval strategies. If you strip away the domain specifics, a learner agent knowledge base looks exactly like this: ingest documents, retrieve relevant context, generate grounded answers, track citations. Vansh built the whole thing, including auth, file storage on AWS S3, a status machine tracking nine ingestion states, a React/Next.js frontend, and a RAGAS evaluation pipeline.

TradeSmith is an autonomous multi-agent system where four AI agents run on a scheduler, research independently, and take real actions (trades) via typed MCP tools. Each agent has persistent memory across runs, can route between multiple LLM providers (OpenAI, DeepSeek, Gemini, Grok), and logs every decision for auditability. This is the same pattern as always-on autonomous agents. The only difference is the domain.

### Technical Stack Match

The role requires async Python, FastAPI, LangChain and LangGraph, vector databases, and production-quality code. Vansh's AtlasRAG project alone covers every one of these: FastAPI plus Uvicorn, LangGraph for agent orchestration, pgvector with HNSW indexing for semantic search, Celery async workers for background ingestion, Supabase for the database, AWS S3 for storage, Clerk for auth, SSE streaming for real-time responses, and Docker for deployment.

Beyond AtlasRAG, Vansh built this AI persona system as part of the screening assignment itself. It uses the same async FastAPI architecture, adds Redis-backed retrieval caching, a LangGraph StateGraph booking agent with Postgres-backed checkpointing via AsyncPostgresSaver, and a Vapi voice integration. The retrieval pipeline runs hybrid vector plus BM25 search fused with RRF and cross-encoder reranking.

### Research Depth Plus Engineering Breadth

Most candidates have one or the other. Vansh has both.

The Noos Technologies internship shows research depth. Implementing HiDDeN, SteganoGAN, and CAISFormer from their original papers means reading architecture diagrams, understanding loss function design (adversarial loss, reconstruction loss, perceptual loss), debugging training instabilities specific to GANs, and reproducing published results without access to the original code. This is not running a HuggingFace tutorial. It requires understanding why the model works.

The production engineering work (AtlasRAG, TradeSmith, this persona system) shows he can take that same depth and ship it. He is not a researcher who struggles to productionize code, and he is not a web developer who bolts on AI features. He operates in the space where both matter.

### Ownership and Leadership

Vansh leads Zero To One, the product and engineering team at E-Cell IIT Roorkee. The team has 15-plus members spanning product, engineering, and design. They have shipped four student products end to end. He manages the product roadmap, engineering execution, and cross-functional coordination across the team.



This matters because Scaler needs people who can own outcomes, not just tasks. Vansh has run a product team. He knows what it means to be responsible for something shipping.

### IIT Roorkee Pedigree

IIT Roorkee is one of the top five engineering institutions in India. Maintaining a CGPA of 8.009 while simultaneously shipping production AI systems, doing a research internship, and leading a 15-plus member team is a signal of genuine capacity.

### The Meta Signal: This Project

Vansh built this AI persona system specifically for this screening process. It is a production RAG system with hybrid retrieval, a LangGraph booking agent, Vapi voice integration, Redis caching, Postgres-backed state, an evaluation framework, and a deployed frontend. He did not just answer the questions in the assignment. He built a system.

That tells you something about how he approaches problems.

## What Vansh Does Not Have (Honest)

Full-time work experience: he is still a student and will graduate in 2027. Everything on his resume is from internships and self-directed projects.

Experience with LiveKit or self-hosted voice infrastructure: he has Vapi experience (managed voice) but has not worked with open-source voice stacks.

Published research papers: he has industry research internship experience implementing papers, but he has not authored publications himself.

## Summary

Vansh is an upcoming final-year IIT Roorkee student who has spent the last two years building production agentic AI systems. His technical stack matches what Scaler needs. His projects mirror what Scaler is building. He has research depth from his Noos internship and engineering breadth from shipping real products. He graduates in 2027 and is actively looking for AI engineer roles where the AI is the core product.
