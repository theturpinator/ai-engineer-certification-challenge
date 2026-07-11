# api

FastAPI backend for Ask MustangDriver (frontend lives in `web/`, later ticket).

## Setup

```sh
cd api
uv venv
uv pip install -r requirements.txt
```

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
