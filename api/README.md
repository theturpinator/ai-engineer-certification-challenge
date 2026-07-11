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
- `POST /chat` with JSON `{"message": str, "user_id": str, "session_id": str}`
  (`session_id` optional — one per browser visit; omitted requests share a
  `"default"` session) → SSE stream:
  1. `data: {"type": "tool", "name": "search_archive" | "web_search" | "check_recalls"}` —
     one event per tool call the agent makes (may be interleaved with tokens; zero or more)
  2. `data: {"type": "token", "text": "..."}` — one event per token as the model generates
  3. `data: {"type": "citations", "citations": [{"title": "...", "url": "..."}]}` — the
     articles actually retrieved this turn (empty list if no retrieval)
  4. `data: [DONE]`

- `GET /garage/{user_id}` →
  `{"profile": {...}, "instructions": [...], "summaries": [{"summary": str, "date": "YYYY-MM-DD"}, ...]}` —
  everything remembered about a user, summaries recent-first (empty
  structures for unknown users, not 404).

`user_id` is the LangGraph thread id: turns with the same user_id share
conversation history via the Postgres checkpointer.

## Memory

Two agent tools write long-term memory to Postgres, keyed by `user_id`
(the UUID is injected via config, never passed by the model):

- `update_garage` — semantic memory: car year/trim/generation, installed
  mods, wishlist, goals. Partial updates merge (lists append-dedupe,
  scalars overwrite); stored as one jsonb row per user in the `garage` table.
- `update_instructions` — procedural memory: standing answer preferences
  ("keep answers short"), appended one row per instruction in the
  `instructions` table.

Episodic memory: after each turn a background task asks Claude Haiku 4.5
for a rolling 2-3 sentence summary of the session (previous summary + latest
exchange), upserted into the `summaries` table keyed by
(`user_id`, `session_id`). A web app has no reliable end-of-session signal,
so the summary is kept current every turn instead.

The system prompt is assembled per turn: base persona + routing policy +
the user's garage profile + standing instructions + up to 5 recent
past-session summaries (dated, excluding the current session), so writes
from previous turns and visits shape every answer.

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
