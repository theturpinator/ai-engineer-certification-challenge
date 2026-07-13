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
  1. `data: {"type": "tool_start", "name": ...}` — emitted the moment the
     model decides to call a tool (before it runs), so clients can show
     status ("Searching the archive…") instead of bare dots
  2. `data: {"type": "tool", "name": "search_archive" | "web_search" | "check_recalls" | "recommend_products" | "update_garage" | "update_instructions" | "complete_onboarding"}` —
     one event per tool call the agent makes, when its result arrives (may
     be interleaved with tokens; zero or more)
  3. `data: {"type": "token", "text": "..."}` — one event per token as the model generates
  4. `data: {"type": "ad", "product": str, "advertiser": str, "description": str, "image": url, "link": url, "sponsored": true, "deltas": {"power": int, ...} | null}` —
     at most three per turn, only when the agent judged the question product-intent
     and called `recommend_products`; `deltas` carries the nine-stat change
     (five performance + four ownership: style, comfort, safety, reliability)
     for the user's first garage car's generation (null when no car is known).
     `link` is the advertiser's click-through URL with its existing UTM
     parameters (an advertiser-level entry links the sponsor's canonical
     website instead); `image` is the hotlinked creative.
  5. `data: {"type": "ping"}` — keepalive whenever ~10s (`CHAT_PING_SECONDS`)
     pass with nothing else to send; clients treat any event as proof of life
     and may declare the connection dead after ~30s of total silence
  6. `data: {"type": "error"}` — the stream failed server-side; the real
     exception is in the server log only. Always followed by `[DONE]`.
  7. `data: {"type": "citations", "citations": [{"title": "...", "url": "..."}]}` — the
     articles actually retrieved this turn (empty list if no retrieval)
  8. `data: [DONE]`

- `GET /garage/{user_id}` →
  `{"profile": {...}, "instructions": [...], "summaries": [{"summary": str, "date": "YYYY-MM-DD"}, ...]}` —
  everything remembered about a user, summaries recent-first (empty
  structures for unknown users, not 404). The profile shape:

  ```json
  {
    "cars": [{
      "id": "8-char id",
      "year": 2016, "trim": "GT", "generation": "S550",
      "color": "...", "nickname": "...",
      "mods": ["..."], "wishlist": ["..."],
      "stats": {"power": 78, "acceleration": 76, "top_speed": 72,
                "handling": 70, "braking": 72,
                "hp": 435, "zero_to_sixty": 4.3, "top_speed_mph": 155,
                "nhtsa": {"stars": 5, "vehicle": "2021 Ford Mustang 2 DR RWD",
                          "url": "https://www.nhtsa.gov/ratings"}},
      "bars": {"current": {"power": 80, "...": 0}, "dream": {"power": 95, "...": 0}},
      "photo_uploaded": false
    }],
    "goals": ["..."]
  }
  ```

  All car fields optional; `stats` is `null`/absent until the background
  enrichment fills it (arcade-style 0–100 scores + real figures for that
  year/trim, generated once by Sonnet and cached — always the STOCK
  baseline, critically calibrated, validated before caching). `stats.nhtsa`
  carries the car's public NHTSA 5-Star overall rating when one exists
  (2011+ model years); the safety score is anchored to it and the UI links
  the source. `bars` is composed on read, never stored: `current` = stock
  stats + the summed per-generation catalog deltas of recognized installed
  mods, `dream` = current + wishlist deltas, both clamped 0–100.
  Mods/wishlist entries resolve to catalog entries by normalized name/alias
  matching; unrecognized free text contributes zero. `photo_uploaded` is true
  once the owner has PUT their own photo (the UI hides the upload pill then).
  Legacy flat single-car profiles are migrated into `cars[0]` transparently
  on read.

- `POST /garage/{user_id}/cars` with JSON
  `{"year": int, "trim": str, "color": str, "nickname": str?}` → 201 + the
  created car (generation derived from the year, enrichment queued in the
  background). 409 when the same year+trim is already in the garage; 400
  `"Garage is full — max 10 cars"` at the 10-car cap (also enforced on the
  chat agent's add-car path; existing over-cap garages are grandfathered).

- `GET /garage/{user_id}/cars/{car_id}/shop` → the car's Upgrade Shop:

  ```json
  {
    "recommended": [{ "...": "2-3 eligible sponsor products, best stat fit" }],
    "catalog": [{
      "id": "tremec-tremec-tkx-5-speed-manual-transmission",
      "name": "...", "advertiser": "Tremec or null", "sponsored": true,
      "description": "...", "categories": ["transmission"],
      "image": "creative URL or null", "link": "UTM click-through or null",
      "deltas": {"power": 0, "acceleration": 2, "...": 0},
      "installed": false, "wishlisted": false
    }]
  }
  ```

  `catalog` is every recommendable sponsor product plus the unbranded
  generic mod categories (no advertiser, no link); `deltas` are resolved for
  this car's generation (null when unknown). The recommended strip excludes
  products already installed/wishlisted and prefers categories the build
  doesn't cover yet. "I have this" / "Add to wishlist" write through the
  normal car PATCH below — this route only reads.

- `GET /garage/{user_id}/cars/{car_id}/image` → the car's portrait (the
  stored content type, `image/png` for generated ones; `Cache-Control:
  public, max-age=86400`); 404 while generation is still pending. Portraits
  are photorealistic renders generated in the background by
  `openai/gpt-image-1` via the gateway, cached in the `car_images` table.
  Each car's portrait is generated exactly once — at creation — then frozen
  forever; no later edit of any kind triggers another image-model call.

- `PUT /garage/{user_id}/cars/{car_id}/image` with raw image bytes in the
  body (`Content-Type: image/*`, ≤8 MB) → 204; replaces the portrait with
  the user's own photo. An uploaded photo is canonical — background
  enrichment never overwrites it; re-uploading replaces it. 415 on
  non-image content, 413 over the size cap.

- `PATCH /garage/{user_id}/cars/{car_id}` with any subset of
  `{year, trim, generation, color, nickname, mods, wishlist}` → the updated
  car (with composed `bars`). UI editing path: scalars overwrite (`null`
  clears), `mods`/`wishlist` replace wholesale (deduped order-preserving,
  case-insensitive, so a retried add can't double-store). Validation: year
  1964–2027, strings length-capped. Identity changes reset `stats` (the
  stock baseline recomputes in the background); mods changes recompose the
  bars deterministically (no LLM). The portrait never changes — frozen after
  its one-time seed.

- `DELETE /garage/{user_id}/cars/{car_id}` → 204; removes the car and its
  cached portrait.

**First-run onboarding (issue #46)** happens inside `POST /chat`, not on a
dedicated endpoint. For a user with no profile at all (no cars, no goals,
no `onboarded` key), the client sends the exact message
`"[begin onboarding]"`; the server stamps `profile.onboarded = false` and,
while the flag is false, injects an interview script into the system
prompt — the agent asks the profile questions one at a time (what to call
them, their Mustang, planned upgrades, future purchase, shows/events,
track days, and how they like answers formatted), records answers with
`update_garage` and `update_instructions`, then calls the
`complete_onboarding` tool, which sets `profile.onboarded = true` (the
client's cue to unlock navigation) and thanks the user. The sentinel message is hidden from
transcript replays and never becomes a chat title. `GET /garage/{user_id}`
exposes the flag via `profile.onboarded`: `false` = interview in progress,
`true` = done, absent = pre-onboarding user (grandfathered when their
profile is non-empty).

`user_id` is the LangGraph thread id: turns with the same user_id share
conversation history via the Postgres checkpointer.

## Memory

Two agent tools write long-term memory to Postgres, keyed by `user_id`
(the UUID is injected via config, never passed by the model):

- `update_garage` — semantic memory: the user's cars (year/trim/generation/
  color/nickname with per-car mods and wishlist) plus user-level goals. The
  garage holds multiple cars: the model passes `car` with the user's own
  words ("my 2016 GT", "the Fox-body"); the backend matches it against
  existing cars (id, then fuzzy identity tokens) and creates a new entry
  when identifying info matches nothing. Partial updates merge (lists
  append-dedupe, scalars overwrite); `remove_mods`/`remove_wishlist` delete
  list items by normalized whole-phrase match ("the cold air intake"
  removes "Cold air intake"), so wishlist→installed moves and
  replaced-X-with-Y swaps land as remove+add in one call, and unmatched
  removals are reported back to the model instead of silently ignored.
  Stored as one jsonb row per user in the `garage` table. After each turn
  a background task fills in missing car stats and portraits.
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

### Ads ingestion

Turns `../data/ads.csv` (the Webflow advertiser export, gitignored) into the
committed product catalog in `ads_artifact/` — `catalog.jsonl` (sponsor
products + generic mod categories, each with per-Mustang-generation stat
deltas, aliases, creative URL, and UTM click-through link) and `vectors.npz`
(row-aligned embeddings for the runtime hybrid index). Only website-active
advertisers are ingested; each one's creatives are vision-analyzed by Sonnet
to classify it (product vendor / service / event / charity / giveaway /
placeholder) and extract products. Recommendation eligibility = active AND
product-or-service vendor: the AdSense placeholder, charities, and giveaways
are ingested but never recommended. Re-running against a fresh export is the
roster-refresh path.

```sh
uv run python -m ingest_ads
```

## Tests

```sh
uv run pytest
```
