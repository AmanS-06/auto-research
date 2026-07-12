# Autonomous Research Pipeline

A multi-agent research system built with **FastAPI**, **LangGraph**, **PostgreSQL**, and **Next.js**. Accepts a research question, breaks it into tasks, searches the web, fact-checks sources, and generates a cited Markdown report.

---

## Architecture

```
                              +-----------------------------------------+
                              |            LangGraph StateGraph        |
                              |                                         |
   POST /api/v1/research ---> | Planner -> Researcher -> Fact-Checker   |
                              |                              -> Writer  |
                              |     |                                      |
                              |     +-- short-circuits to END on error    |
                              +---------------------+--------------------+
                                                    |
                                                    v
                       +------------------------------------------+
                       |  ResearchService (app/services)           |
                       |  - final_state["report_markdown"] --->    |
                       |  - persist ResearchReport                 |
                       |  - mark job complete / failed             |
                       |  - stream progress per graph node         |
                       +--------------------+---------------------+
                                            |
                           +-----------------+------------------+
                           v                                    v
              +-------------------+              +------------------------+
              |  PostgreSQL       |<-- polled -->|  GET /api/v1/research  |
              |  - research_jobs  |              |  GET .../status        |
              |  - research_      |              |                        |
              |    reports        |              |                        |
              +-------------------+              +------------------------+
```

### Module Map

| Layer    | Module                                                       | Owns                                         |
|----------|--------------------------------------------------------------|----------------------------------------------|
| API      | `app/api/v1/research.py`                                    | HTTP routes, background task dispatch        |
| Service  | `app/services/research_service.py`                          | Job lifecycle, persistence, progress stream  |
| Service  | `app/services/checkpoint.py`                                | LangGraph Postgres checkpointer              |
| Graph    | `app/core/langgraph/graph.py`                               | StateGraph wiring                            |
| Graph    | `app/core/langgraph/state.py`                               | `ResearchState` TypedDict (single source)    |
| Graph    | `app/core/langgraph/nodes/{planner,researcher,fact_checker,writer}.py` | Agent nodes                      |
| Search   | `app/core/langgraph/tools/web_search.py`                    | Async Serper client                          |
| LLM      | `app/core/llm.py`                                           | OpenAI-compatible chat-model factory         |
| Persistence | `app/models/research.py`                                 | `ResearchJob`, `ResearchReport` SQLModels    |
| Schemas  | `app/schemas/{research,evidence}.py`                        | Request/response models                      |

---

## Tech Stack

- **API**: FastAPI (async)
- **Orchestration**: LangGraph with PostgreSQL checkpointer
- **LLM**: Any OpenAI-compatible endpoint (default: OpenRouter with NVIDIA Nemotron 3 Ultra free)
- **Search**: Serper API (Google Search)
- **Database**: PostgreSQL + SQLModel
- **Containerization**: Docker + Docker Compose
- **Frontend**: Next.js 16 + React 19 + Tailwind v4 (in `frontend/`)

> **OpenRouter / Nemotron 3 Ultra (free) caveat.** OpenRouter's Models API currently advertises `tools` / `tool_choice` for `nvidia/nemotron-3-ultra-550b-a55b:free` but does **not** advertise `response_format` / `structured_outputs`. Our agents use LangChain's `with_structured_output()` which the OpenAI Python client may map to either tool calling or `response_format` depending on the schema. If you see `Pydantic validation` or `JSONDecodeError` coming back from the LLM, switch to a model that advertises `structured_outputs` (e.g. the paid Nemotron 3 Ultra entry, or `openai/gpt-4o-mini`).

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- An OpenAI-compatible API key (OpenRouter / OpenAI / DeepSeek / ...)
- Serper API key

### Configuration

```bash
cp .env.example .env
# Edit .env with your keys
```

**Required environment variables:**

| Variable           | Purpose                                                |
|--------------------|--------------------------------------------------------|
| `LLM_API_KEY`      | Your OpenAI-compatible API key                         |
| `LLM_BASE_URL`     | Endpoint base URL (default: OpenRouter)                |
| `LLM_MODEL`        | Model name (default: Nemotron 3 Ultra free on OpenRouter) |
| `SERPER_API_KEY`   | Serper API key                                         |
| `DATABASE_URL`     | PostgreSQL asyncpg DSN                                 |

### Run with Docker

```bash
docker-compose up -d
```

API available at `http://localhost:8000`. Migrations are applied via the canonical `alembic/versions/0001_initial.py`.

### Local Development

```bash
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

---

## API Usage

### Start Research

```bash
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the latest developments in quantum computing?",
    "max_tasks": 5,
    "max_sources_per_task": 3
  }'
```

`max_tasks` is clamped to `[1, 25]`; `max_sources_per_task` to `[1, 10]`. Out-of-range values return 422.

**Response (HTTP 202):**
```json
{
  "job_id": "uuid",
  "status": "pending"
}
```

### Check Status

```bash
curl http://localhost:8000/api/v1/research/{job_id}/status
```

```json
{
  "job_id": "uuid",
  "status": "running",
  "progress": "Researching the web...",
  "error": null
}
```

`progress` updates as the graph advances:
`Planning research tasks...` → `Researching the web...` → `Fact-checking sources...` → `Writing report...` → `Research complete`

### Get Report

```bash
curl http://localhost:8000/api/v1/research/{job_id}
```

**Response (when `status == "complete"`):**
```json
{
  "job_id": "uuid",
  "status": "complete",
  "report": "# Executive Summary\n...\n## References\n[1] Title. URL\n[2] Title. URL",
  "citations": [
    {"id": "1", "source_url": "...", "title": "...", "snippet": "..."}
  ]
}
```

---

## State and Persistence Contracts

The graph's final state is the single source of truth for what becomes a report:

```python
final_state = {
    "status": "complete" | "error",
    "error": str | None,
    "research_tasks": list[ResearchTask],
    "evidence": list[Evidence],
    "verified_evidence": list[Evidence],
    "report_markdown": str,
    "summary": str,
    "citations": list[Citation],
}
```

`app.services.research_service.extract_report_payload(final_state)` is the canonical helper for reading this. It returns a JSON-safe dict (Pydantic models are recursively `model_dump(mode="json")`-ed) so the result is safe to hand to a SQLAlchemy `JSON`/`JSONB` column.

The service persists:

```python
ResearchReport(
    job_id=job_id,
    question=request.question,
    report=final_state["report_markdown"],
    citations=final_state["citations"],
    extra_metadata={
        "summary": final_state["summary"],
        "tasks_completed": ...,
        "sources_found": ...,
        "verified_sources": ...,
        "citation_count": ...,
    },
)
```

Job lifecycle in the DB is high-level: `pending` → `running` → (`complete` | `failed`). The graph's finer phases (`planning` / `researching` / `fact_checking` / `writing`) are reflected in `job.progress` (a human-readable string) and the final graph state is serialized into `job.state` on completion.

---

## Project Structure

```
app/
+-- api/v1/                # FastAPI endpoints
+-- core/
|   +-- config.py          # Pydantic Settings
|   +-- database.py        # Async SQLModel setup
|   +-- llm.py             # OpenAI-compatible chat-model factory
|   +-- middleware.py      # CORS + request logging
|   +-- citations.py       # Citation formatting & dedup helpers
|   +-- langgraph/
|       +-- state.py       # ResearchState TypedDict
|       +-- graph.py       # StateGraph builder
|       +-- nodes/         # Agent nodes
|           +-- planner.py
|           +-- researcher.py
|           +-- fact_checker.py
|           +-- writer.py
+-- models/                # SQLModel tables
+-- schemas/               # Pydantic request/response
+-- services/
    +-- checkpoint.py      # LangGraph Postgres checkpointer
    +-- research_service.py  # Pipeline orchestration
```

---

## Environment Variables

| Variable                 | Description                                  | Required | Default                                            |
|--------------------------|----------------------------------------------|----------|----------------------------------------------------|
| `LLM_API_KEY`            | OpenAI-compatible API key                    | Yes      | (empty)                                            |
| `LLM_BASE_URL`           | API base URL                                 | No       | `https://integrate.api.nvidia.com/v1`                     |
| `LLM_MODEL`              | Model name                                   | No       | `deepseek-ai/deepseek-v4-flash`           |
| `LLM_TEMPERATURE`        | Sampling temperature                         | No       | `0.2`                                              |
| `LLM_MAX_TOKENS`         | Max output tokens                            | No       | `4096`                                             |
| `LLM_TIMEOUT_SECONDS`    | Per-request timeout                          | No       | `60`                                               |
| `SERPER_API_KEY`         | Serper API key                               | Yes      | (empty)                                            |
| `SERPER_BASE_URL`        | Serper base URL                              | No       | `https://google.serper.dev`                        |
| `SERPER_TIMEOUT_SECONDS` | Serper timeout                               | No       | `20`                                               |
| `DATABASE_URL`           | PostgreSQL asyncpg DSN                       | Yes      | (empty)                                            |
| `APP_ENV`                | Environment (development/production)         | No       | `development`                                      |
| `LOG_LEVEL`              | Logging level                                | No       | `INFO`                                             |
| `MAX_RESEARCH_TASKS`     | Hard cap on planner tasks                    | No       | `5`                                                |
| `MAX_SOURCES_PER_TASK`   | Hard cap on evidence per task                | No       | `3`                                                |
| `MIN_SOURCE_RELEVANCE`   | Floor for fact-checker                       | No       | `0.4`                                              |

---

## Testing

```bash
pytest tests/ -v
```

**Test coverage:**

- `tests/test_agents/` — each agent node + end-to-end graph with fake LLM/Serper
- `tests/test_research_service.py` — service layer (success, error, fallback, JSON round-trip, progress streaming)
- `tests/test_api.py` — FastAPI routes with mocked DB session
- `tests/test_database.py` — canonical migration matches models
- `tests/test_schemas.py` — Pydantic validation including request limits
- `tests/test_web_search.py` — Serper client (parsing, retries, malformed responses)
- `tests/test_imports.py` — module surface
- `tests/integration/test_pipeline.py` — pipeline-level smoke + citation utilities

---

## Known Limitations

- **Job cancellation** — once submitted, a research job cannot be cancelled.
- **Job status granularity** — the DB job status is high-level (`pending`/`running`/`complete`/`failed`). In-flight graph phase is reflected as a human-readable string in `job.progress` and updated per node.
- **Concurrent job cap** — there is no rate limiting; a client can submit unlimited concurrent jobs.
- **No streaming** — clients must poll `GET /research/{id}` or `GET /research/{id}/status` for progress. WebSocket / SSE is on the roadmap.
- **Structured-output reliability on free models** — the default `nvidia/nemotron-3-ultra-550b-a55b:free` model does not advertise `response_format` / `structured_outputs` in OpenRouter's Models API. Agents that depend on `with_structured_output()` may intermittently fail. Switch to a model that advertises those parameters for production.
- **LaTeX/PDF export removed** — the earlier `app/core/latex.py` and `app/tools/web_search.py` orphans were deleted for the MVP. Adding PDF export later should live behind a clean `app/export/` module boundary.
- `uv.lock` not committed — the production `Dockerfile` uses `pip install -e .` rather than `uv pip install` because the lock file is not yet present. See `Dockerfile` for the current install path.

---

## Future Enhancements

- Streaming progress via WebSocket / Server-Sent Events
- Job cancellation
- User authentication & job ownership
- Rate limiting on `POST /research`
- LaTeX/PDF export (clean module boundary)
- Scheduled/recurring research jobs
- Source credibility ML model
- Multi-language support

---

## License

MIT