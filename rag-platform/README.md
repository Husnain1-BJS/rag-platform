# Threat Intelligence RAG Platform

A cybersecurity threat intelligence platform using RAG (Retrieval-Augmented Generation) with FastAPI, Qdrant, and Sentence Transformers.

## Features

- **Hybrid Search**: Vector + BM25 sparse search with cross-encoder re-ranking
- **Data Sources**: NVD CVEs (local/API) + MITRE ATT&CK techniques
- **Observability**: Structured JSON logging, Prometheus metrics, OpenTelemetry tracing
- **Auth & Rate Limiting**: API key authentication + token bucket rate limiting
- **Pipeline**: Async embedding, connection pooling, deduplication, checkpoint/resume

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- OpenRouter API key (for LLM)

### Installation

```bash
# Clone repo
git clone https://github.com/Husnain1-BJS/rag-platform.git
cd rag-platform

# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY
```

### Run Locally

```bash
# Terminal 1: Start API server
poetry run uvicorn apps.api.main:app --reload --port 8000

# Terminal 2: Ingest data (NVD recent + MITRE)
poetry run python -m apps.ingestion.pipeline --source all --limit 200

# Or incremental sync (last 7 days)
poetry run python -m apps.ingestion.pipeline --source nvd-api --limit 0 --days-back 7
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-key-if-configured" \
  -d '{"question": "What is CVE-2024-1234?", "search_type": "hybrid", "top_k": 5}'
```

### Health & Metrics

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

## Docker

```bash
docker-compose -f infra/docker/docker-compose.yml up --build
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check (Qdrant + LLM) |
| GET | `/metrics` | No | Prometheus metrics |
| POST | `/query` | Yes | RAG query |
| POST | `/ingest` | Yes | Trigger ingestion |

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | - | Required for LLM |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Free model |
| `QDRANT_PATH` | `./qdrant_data` | Local Qdrant storage |
| `API_KEY` | `` | Empty = auth disabled |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests per IP per minute |
| `ENABLE_RERANKING` | `true` | Cross-encoder re-ranking |
| `ENABLE_HYBRID_SEARCH` | `true` | BM25 + vector |

## Tests

```bash
poetry run pytest tests/ -v
```

## Project Structure

```
apps/
  api/           # FastAPI application
    main.py      # API routes + lifespan
    config.py    # Pydantic settings
    auth.py      # API key middleware
    rate_limit.py# Token bucket limiter
    logging_config.py
    metrics.py   # Prometheus metrics
    tracing.py   # OpenTelemetry
  ingestion/     # Data pipeline
    pipeline.py  # Orchestration
    embedder.py  # Async embeddings
    indexer.py   # Qdrant upsert + dedup
    chunker.py   # Text splitting
    parsers/     # CVE + MITRE extraction
    fetchers/    # NVD REST API client
tests/
  unit/          # 32 unit tests
  integration/   # 6 API tests
.github/workflows/tests.yml  # CI pipeline
```

## License

MIT