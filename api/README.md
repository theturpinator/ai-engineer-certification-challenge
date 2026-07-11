# api

FastAPI backend for Ask MustangDriver (frontend lives in `web/`, later ticket).

## Setup

```sh
cd api
uv venv
uv pip install -r requirements.txt
```

## Running the chat API

Needs the repo-root `.env` (AI_GATEWAY_API_KEY, DATABASE_URL, LANGSMITH_*)
and the local Postgres from the repo-root compose file:

```sh
docker compose up -d          # from the repo root (Postgres on host port 5433)
cd api
uv run uvicorn app:app --port 8000
```

- `GET /health` → `{"status": "ok"}`
- `POST /chat` with JSON `{"message": str, "user_id": str}` → SSE stream:
  1. `data: {"type": "token", "text": "..."}` — one event per token as the model generates
  2. `data: {"type": "citations", "citations": [{"title": "...", "url": "..."}]}` — the
     articles actually retrieved this turn (empty list if no retrieval)
  3. `data: [DONE]`

`user_id` is the LangGraph thread id: turns with the same user_id share
conversation history via the Postgres checkpointer.

## Ingestion

Turns `../data/articles-clean.csv` (gitignored) into the committed index
artifact in `index_artifact/` — `chunks.jsonl` (chunk text + metadata) and
`vectors.npz` (float32 text-embedding-3-small vectors, row-aligned with the
JSONL). Needs `AI_GATEWAY_API_KEY` in the repo-root `.env`.

```sh
uv run python -m ingest
```

Pipeline details are in `ingest.py`'s docstring; the 21 excluded promo
articles are documented in `EXCLUDED_ARTICLES.md`.

## Tests

```sh
uv run pytest
```
