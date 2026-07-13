# Agent guide — Ask MustangDriver

Agentic RAG chatbot for MustangDriver.com. The spec is [GitHub issue #1](https://github.com/theturpinator/ai-engineer-certification-challenge/issues/1); every increment ships from a `ready-for-agent` issue. The API contract lives in `api/app.py`'s module docstring and `api/README.md` — keep both current when the contract changes.

## Workflow

- **Issue-first:** file (or point at) a GitHub issue before implementing any change.
- **Ship gate:** every increment passes show-first review — local build + screenshots + owner approval — **before** any production deploy.
- **Test seams:** seam 1 = HTTP boundary (real Postgres, real LLM where routing is the behavior); seam 2 = pure functions (no network). LLM/vision/image calls stay behind the pure/impure split (see `api/ingest.py`, `api/ingest_ads.py`).

## Commands

| What | How |
|---|---|
| Local stack | `docker compose up -d` (Postgres :5433), then `cd api && uv run uvicorn app:app --port 8000` and `cd web && npm run dev` (Node 18+) |
| API tests | `cd api && uv run pytest` — needs the repo-root `.env` and the compose Postgres; seam-1 suites call the real LLM |
| Web typecheck/build | `cd web && npx tsc --noEmit && npm run build` |
| Web e2e (Playwright) | `cd web && npm run e2e` — needs the compose Postgres; starts (or reuses) the API and web dev servers; records a video per test under `web/test-results/` (Node 18+) |
| Article ingestion | `cd api && uv run python -m ingest` (reads `data/articles-clean.csv`, gitignored) |
| Ads ingestion | `cd api && uv run python -m ingest_ads` (reads `data/ads.csv`, gitignored) — the roster-refresh path |
| **Production deploy** | `./deploy.sh` from the repo root — api → health check → web → smoke check, fail-fast. Only after the ship gate. |

## Hard rules

- Raw CSVs live in `data/` (gitignored) — never commit them. The derived artifacts (`api/index_artifact/`, `api/ads_artifact/`) **are** committed.
- The article index, archive citations, and published eval results (`evals/results/`) are graded artifacts — features must not touch them.
- Sponsored content is always labeled Sponsored, capped at two cards per chat turn, and only recommendation-eligible advertisers (active product/service vendors) are ever recommended.
