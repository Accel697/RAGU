# FastAPI Demo

This demo runs a GraphRAG-style API service with:

- **grg-demo**: FastAPI app from `server.py`
- **knb-demo**: Apache AGE backend for graph storage
- **vllm-demo**: Ollama server providing OpenAI-compatible LLM and embeddings endpoints (CPU-optimized)

The stack is defined in `examples/fastapi_demo/docker-compose.yml`.

## What This Demo Exposes

The service listens on `http://127.0.0.1:8000` and provides:

- **GET `/graph/records`** - retrieve all nodes and edges currently stored in the graph database
- **POST `/graph/load`** - ingest text files from `/app/data/ru`, build knowledge graph, and sync to Apache AGE
- **POST `/query`** - local graph-aware search with Russian language response generation

## Prerequisites

- Docker + Docker Compose
- CPU-only environment

## Run

From repository root:

```bash
docker compose -f examples/fastapi_demo/docker-compose.yml up --build
```

When startup is complete, you should see the API at:

- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`