"""Ask MustangDriver chat API.

FastAPI app: POST /chat streams SSE tokens + tool/citations/ad events from a
LangGraph ReAct agent with four tools: search_archive (in-memory Qdrant over
index_artifact/, built at startup), web_search (Tavily, live web),
check_recalls (NHTSA Recalls API), and recommend_products (intent-gated
sponsored recommendations over the committed ads_artifact/ catalog via a
mini BM25+dense RRF index; at most two `ad` events per turn, each carrying
the creative image, UTM click-through link, Sponsored flag, and stat-delta
chips for the user's car), plus two memory tools: update_garage (semantic:
the user's cars, per-car mods/wishlist, user-level goals) and
update_instructions (procedural: standing answer preferences), both keyed by
user_id in Postgres and folded into the system prompt each turn. Episodic
memory: after each turn a background task has Claude Haiku keep a rolling 2-3
sentence summary of the session (keyed by user_id + session_id from the
client), and recent past-session summaries are injected into the system
prompt. The garage holds multiple cars (profile.cars, legacy flat profiles
migrate on read); each car is enriched in the background with arcade-style
STOCK-baseline stats (Sonnet, cached in the car with an identity
fingerprint) and an AI-generated portrait (gpt-image-1 via the gateway,
cached in car_images with identity/build fingerprints). The portrait is
canonical: identity changes (year/generation/trim) regenerate it; build
changes (color/mods) EDIT the stored photo via gemini-2.5-flash-image, never
re-rolling. Bars compose deterministically from the committed catalog:
current = stock + summed deltas of recognized installed mods, dream =
current + wishlist deltas, both clamped 0-100 and returned by the garage
read/write endpoints so clients never re-derive them. GET
/garage/{user_id}/cars/{car_id}/shop serves the Upgrade Shop (recommended
sponsor products + the full searchable catalog with per-car delta chips);
its have-it/want-it actions write through the existing car PATCH. GET
/garage/{user_id} returns everything known; cars are editable via
PATCH/DELETE /garage/{user_id}/cars/{car_id}. Users can hold multiple chats:
POST /chat takes an optional chat_id (thread_id = user_id:chat_id, the legacy
"default" chat keeps the bare user_id thread), GET /chats/{user_id} lists
them, GET /chats/{user_id}/{chat_id}/messages replays a transcript from the
Postgres checkpointer (docker compose, repo root). Login is optional: POST
/auth/google exchanges a Google ID token for an app JWT bound to a canonical
user_id (identities table, google_sub -> user_id; first login adopts the
browser's anonymous id), and a valid Authorization bearer overrides the
path/body user id on every user-scoped route.

Run locally:  uv run uvicorn app:app --port 8000
"""

import asyncio
import base64
import json
import os
import re
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import httpx
import jwt
import numpy as np
import psycopg
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from psycopg.types.json import Json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, StringConstraints
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi

from ingest_ads import STATS as BAR_STATS, embed_text

API_DIR = Path(__file__).parent
load_dotenv(API_DIR.parent / ".env")  # before any graph runs, so LangSmith traces

GATEWAY_URL = "https://ai-gateway.vercel.sh/v1"
COLLECTION = "archive"
TOP_K = 5

SYSTEM_PROMPT = """You are the Ask MustangDriver assistant, an enthusiastic and \
knowledgeable guide to the MustangDriver.com article archive.

Tool policy:
- search_archive first for Mustang history, specs, builds, reviews, and lore. \
Ground those answers in the retrieved articles and cite sources inline as \
markdown links using each article's title and URL, e.g. \
[Article Title](https://www.mustangdriver.com/...).
- web_search only when the archive comes up empty or the question is \
inherently live (current prices, market values, news, upcoming events, \
availability). Any answer built on web results MUST begin with "According to \
a live web search" and cite the source pages.
- check_recalls for safety-recall questions; report the official NHTSA \
campaigns (component, summary, remedy, date), or that none were found.
- recommend_products ONLY when the user is shopping for a part or upgrade, \
asks for upgrade advice, or the answer naturally calls for a specific \
product. Call it at most once per turn. Weave the fitting product(s) into \
your answer, naming the advertiser behind each. NEVER call it on history, \
spec, lore, recall, news, or any question that doesn't call for buying \
something — those answers stay purely editorial, exactly as before.

If no tool can answer, say so plainly rather than guessing.

Memory policy:
- Whenever the user mentions facts about one of their own cars — model year, \
trim, generation (e.g. S550), color, nickname, installed mods, \
wishlist/planned mods, or goals (track, show, daily driver) — silently call \
update_garage with those facts. Record the car IMMEDIATELY, on its FIRST \
mention, with whatever is known so far — partial info is expected: a trim \
alone, or a color plus model, is enough ("I have a blue Mustang GT" means \
you call update_garage right away with trim GT and color blue). NEVER wait \
for complete details like the model year before recording; you may ask a \
follow-up question, but record first, then update the SAME car as more \
details arrive in later turns. The garage holds multiple cars: pass the \
`car` argument with the user's own words for which car they mean (e.g. \
"my 2016 GT", "the Fox-body"). When the user mentions a car of theirs that \
is NOT yet in their garage profile, record it as a new garage entry by \
calling update_garage with its facts. Never announce or mention that you \
recorded anything.
- Whenever the user states a standing preference about how you should answer \
(e.g. "keep answers short", "always end with X"), silently call \
update_instructions with that preference, then follow it.
- Always follow every standing instruction listed below, in every answer."""

_embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url=GATEWAY_URL,
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    check_embedding_ctx_length=False,  # gateway wants raw strings, not token arrays
)

_summarizer = ChatOpenAI(
    model="anthropic/claude-haiku-4.5",
    base_url=GATEWAY_URL,
    api_key=os.environ["AI_GATEWAY_API_KEY"],
)

_stats_llm = ChatOpenAI(
    model="anthropic/claude-sonnet-4.5",
    base_url=GATEWAY_URL,
    api_key=os.environ["AI_GATEWAY_API_KEY"],
)

_qdrant: QdrantClient | None = None
_agent = None


def build_index() -> QdrantClient:
    """In-memory Qdrant collection from the committed index artifact."""
    with open(API_DIR / "index_artifact" / "chunks.jsonl") as f:
        chunks = [json.loads(line) for line in f]
    vectors = np.load(API_DIR / "index_artifact" / "vectors.npz")["vectors"]
    client = QdrantClient(location=":memory:")
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    client.upload_collection(
        COLLECTION, vectors=vectors, payload=chunks, ids=list(range(len(chunks)))
    )
    return client


@tool
async def search_archive(query: str) -> str:
    """Search the MustangDriver article archive. Returns a JSON list of the most
    relevant excerpts, each with the article's title, url, and text."""
    vector = await _embeddings.aembed_query(query)
    hits = _qdrant.query_points(COLLECTION, query=vector, limit=TOP_K).points
    return json.dumps(
        [{"title": h.payload["title"], "url": h.payload["url"], "text": h.payload["text"]}
         for h in hits],
        ensure_ascii=False,
    )


@tool
async def web_search(query: str) -> str:
    """Search the live web (Tavily) for current information the archive can't
    answer: prices, market values, news, events, availability. Returns a JSON
    list of results, each with title, url, and content snippet."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "max_results": 5,
            },
            timeout=30,
        )
    resp.raise_for_status()
    return json.dumps(
        [{"title": r["title"], "url": r["url"], "content": r["content"]}
         for r in resp.json()["results"]],
        ensure_ascii=False,
    )


@tool
async def check_recalls(year: int, make: str = "Ford", model: str = "Mustang") -> str:
    """Look up official NHTSA safety recalls for a vehicle model year
    (defaults to Ford Mustang). Returns a JSON list of recall campaigns with
    component, summary, remedy, and report date, or a no-recalls message."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.nhtsa.gov/recalls/recallsByVehicle",
            params={"make": make, "model": model, "modelYear": year},
            timeout=30,
        )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return f"No NHTSA recalls found for the {year} {make} {model}."
    return json.dumps(
        [{"campaign": r["NHTSACampaignNumber"], "date": r["ReportReceivedDate"],
          "component": r["Component"], "summary": r["Summary"], "remedy": r["Remedy"]}
         for r in results],
        ensure_ascii=False,
    )


# ponytail: connection-per-call; pool if throughput ever matters
async def _db() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(
        os.environ["DATABASE_URL"], autocommit=True
    )


LEGACY_CAR_FIELDS = ("year", "trim", "generation", "mods", "wishlist")
CAR_MATCH_FIELDS = ("year", "trim", "generation", "nickname", "color")


def _migrate(profile: dict) -> dict:
    """Wrap legacy flat single-car fields into cars[0] (in place); idempotent."""
    if any(k in profile for k in LEGACY_CAR_FIELDS):
        car = {"id": uuid.uuid4().hex[:8]}
        for k in LEGACY_CAR_FIELDS:
            if k in profile:
                car[k] = profile.pop(k)
        profile["cars"] = [car] + profile.get("cars", [])
    return profile


def _car_tokens(*values) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", " ".join(str(v) for v in values).lower()))


def _conflicts(car: dict, updates: dict) -> bool:
    """True when updates name a different car (an identity field disagrees)."""
    for k in ("year", "trim", "generation"):
        if updates.get(k) and car.get(k):
            a, b = str(updates[k]).lower(), str(car[k]).lower()
            if a != b if k == "year" else (a not in b and b not in a):
                return True
    return False


def _match_car(cars: list[dict], desc: str | None, updates: dict) -> dict | None:
    """The existing car a garage update targets, or None to create a new one.

    Order: exact id match on `desc`; fuzzy token overlap of `desc` against each
    car's identity fields; identity fields in the updates themselves; a lone
    car when nothing disagrees. Identifying info that matches nothing => None
    (the caller creates a new car)."""
    if desc:
        for c in cars:
            if c.get("id") == desc:
                return c
        want = _car_tokens(desc)
        score, best = max(
            ((len(want & _car_tokens(*(c.get(k, "") for k in CAR_MATCH_FIELDS))), c)
             for c in cars),
            default=(0, None), key=lambda t: t[0],
        )
        return best if best is not None and score > 0 and not _conflicts(best, updates) else None
    for c in cars:
        agree = any(
            updates.get(k) and c.get(k)
            and (str(updates[k]) == str(c[k]) if k == "year"
                 else str(updates[k]).lower() in str(c[k]).lower()
                 or str(c[k]).lower() in str(updates[k]).lower())
            for k in ("year", "trim", "generation")
        )
        if agree and not _conflicts(c, updates):
            return c
    if len(cars) == 1 and not _conflicts(cars[0], updates):
        return cars[0]
    if any(updates.get(k) for k in ("year", "trim", "generation", "color")):
        return None  # identifying info (trim/color count) matching no car -> new entry
    return cars[0] if cars else None  # ponytail: ambiguous target defaults to first car


async def _load_profile(conn, user_id: str) -> dict:
    """Garage profile, migrated to the cars[] shape. A legacy flat profile is
    written back once so the migrated car id is stable across reads."""
    row = await (
        await conn.execute("SELECT profile FROM garage WHERE user_id = %s", (user_id,))
    ).fetchone()
    raw = row[0] if row else {}
    profile = _migrate(dict(raw))
    if profile != raw:
        await conn.execute(
            "INSERT INTO garage (user_id, profile) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET profile = EXCLUDED.profile",
            (user_id, Json(profile)),
        )
    return profile


async def _get_memory(user_id: str) -> tuple[dict, list[str], list[dict]]:
    """(garage profile, standing instructions, recent session summaries) for a
    user; empty when unknown. Summaries are recent-first, with session_id so
    /chat can exclude the session in progress (limit 6 = 5 past + current)."""
    async with await _db() as conn:
        profile = await _load_profile(conn, user_id)
        rows = await (
            await conn.execute(
                "SELECT instruction FROM instructions WHERE user_id = %s ORDER BY id",
                (user_id,),
            )
        ).fetchall()
        srows = await (
            await conn.execute(
                "SELECT session_id, summary, updated_at FROM summaries "
                "WHERE user_id = %s ORDER BY updated_at DESC LIMIT 6",
                (user_id,),
            )
        ).fetchall()
    summaries = [
        {"session_id": r[0], "summary": r[1], "date": str(r[2].date())} for r in srows
    ]
    return profile, [r[0] for r in rows], summaries


async def _summarize(user_id: str, session_id: str, user_text: str, parts: list[str]):
    """Keep a rolling 2-3 sentence summary of the session, upserted after every
    turn (background task, runs once the SSE stream has been sent). A web app
    has no reliable end-of-session signal, so updating the summary each turn is
    what makes "after the session" true; feeding Haiku the previous summary plus
    the latest exchange avoids replaying the whole multi-session thread."""
    assistant_text = "".join(parts)
    if not assistant_text:
        return
    try:
        async with await _db() as conn:
            row = await (
                await conn.execute(
                    "SELECT summary FROM summaries WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
            ).fetchone()
            prompt = (
                "Write a 2-3 sentence summary of this chat session: the topics "
                "discussed, key facts about the user's car, and any advice given. "
                "Reply with only the summary.\n\n"
                + (f"Session summary so far:\n{row[0]}\n\n" if row else "")
                + f"Latest exchange:\nUser: {user_text}\nAssistant: {assistant_text}"
            )
            resp = await _summarizer.ainvoke(prompt)
            await conn.execute(
                "INSERT INTO summaries (user_id, session_id, summary) VALUES (%s, %s, %s) "
                "ON CONFLICT (user_id, session_id) DO UPDATE "
                "SET summary = EXCLUDED.summary, updated_at = now()",
                (user_id, session_id, resp.content.strip()),
            )
    except Exception as e:  # ponytail: a lost summary beats a failed request
        print(f"session summary failed: {e}")


def _car_desc(car: dict) -> str:
    """'2016 S550 GT Premium' — identity fields only, in a natural order."""
    return " ".join(str(car[k]) for k in ("year", "generation", "trim") if car.get(k))


GENERATIONS = (
    (1964, 1973, "First generation"),  # 1964½ cars are first-gen
    (1974, 1978, "Mustang II"),
    (1979, 1993, "Fox-body"),
    (1994, 2004, "SN95"),
    (2005, 2014, "S197"),
    (2015, 2023, "S550"),
    (2024, 9999, "S650"),
)


def _derive_generation(year) -> str | None:
    """Mustang generation short name for a model year, or None if unmappable."""
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    return next((g for lo, hi, g in GENERATIONS if lo <= y <= hi), None)


def _autofill_generation(car: dict) -> None:
    """Fill in the generation from the year when missing (all merge paths:
    chat tool, PATCH, and picker POST route through this)."""
    if car.get("year") and not car.get("generation"):
        gen = _derive_generation(car["year"])
        if gen:
            car["generation"] = gen


# --- Product catalog (committed by ingest_ads.py): sponsor products + generic
# mod categories, each with per-generation stat deltas. Read-only at runtime;
# every user sees the same numbers every time.

with open(API_DIR / "ads_artifact" / "catalog.jsonl") as f:
    CATALOG = [json.loads(line) for line in f]

_GEN_KEYS = {re.sub(r"[^a-z0-9]", "", g.lower()): g for _lo, _hi, g in GENERATIONS}

# Mini hybrid retrieval (BM25 + dense, RRF — the technique validated in the
# retrieval experiments) over the RECOMMENDABLE catalog entries only: the
# placeholder, charities, and giveaways are in the catalog but never in this
# index. Entirely separate from the article index; no network at startup.
AD_TOP_K = 2  # at most two sponsored cards per turn
_RRF_K = 60
_RECO_ROWS = [i for i, e in enumerate(CATALOG) if e["recommendable"]]
_RECO = [CATALOG[i] for i in _RECO_ROWS]
_RECO_VECTORS = np.load(API_DIR / "ads_artifact" / "vectors.npz")["vectors"][_RECO_ROWS]


def _tokenize(text: str) -> list[str]:
    """Lowercase; keep decimal numbers ("5.0") and alnum runs ("s550") whole."""
    return re.findall(r"\d+(?:\.\d+)+|[a-z0-9]+", text.lower())


_RECO_BM25 = BM25Okapi([_tokenize(embed_text(e)) for e in _RECO]) if _RECO else None
_RECO_BY_ID = {e["id"]: e for e in _RECO}


async def _search_products(query: str) -> list[dict]:
    """Top recommendable products for a query: reciprocal rank fusion of the
    dense and BM25 rankings (k=60), like the archive hybrid retriever."""
    if not _RECO:
        return []
    vector = np.asarray(await _embeddings.aembed_query(query), dtype=np.float32)
    dense_ranking = list(np.argsort(-(_RECO_VECTORS @ vector)))
    scores = _RECO_BM25.get_scores(_tokenize(query))
    bm25_ranking = [i for i in np.argsort(-scores) if scores[i] > 0]
    fused: dict[int, float] = defaultdict(float)
    for ranking in (dense_ranking, bm25_ranking):
        for rank, idx in enumerate(ranking):
            fused[int(idx)] += 1.0 / (_RRF_K + rank + 1)
    top = sorted(fused, key=fused.__getitem__, reverse=True)[:AD_TOP_K]
    return [_RECO[i] for i in top]


@tool
async def recommend_products(query: str) -> str:
    """Search the site's sponsor product catalog for a specific product to
    recommend. Call this ONLY when the user is shopping for a part or
    upgrade, asks for upgrade advice, or the answer naturally calls for a
    product — never on history, spec, lore, recall, or other questions.
    Returns a JSON list of sponsored products (name, advertiser, one-line
    description). Weave the fitting one(s) into your answer, naming the
    advertiser; the chat UI shows each as a Sponsored card automatically."""
    hits = await _search_products(query)
    return json.dumps(
        [{"id": h["id"], "product": h["name"], "advertiser": h["advertiser"],
          "description": h["description"]} for h in hits],
        ensure_ascii=False,
    )


def _gen_key(generation) -> str | None:
    """A car's generation string mapped to its catalog deltas key ("fox body",
    "Fox-body", "foxbody" all resolve); None when unrecognized."""
    return _GEN_KEYS.get(re.sub(r"[^a-z0-9]", "", str(generation or "").lower()))


def _norm_words(text) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _match_entry(text: str, catalog: list[dict] | None = None) -> dict | None:
    """The catalog entry a free-text mod/wishlist item refers to, or None.
    Whole-word name/alias match; the longest (most specific) alias wins, so
    "cold air intake" beats "intake". Unrecognized free text matches nothing
    and simply contributes zero to the bars."""
    haystack = f" {_norm_words(text)} "
    best, best_len = None, 0
    for entry in CATALOG if catalog is None else catalog:
        for alias in (entry["name"], *entry.get("aliases", [])):
            a = _norm_words(alias)
            if len(a) > best_len and f" {a} " in haystack:
                best, best_len = entry, len(a)
    return best


def _sum_deltas(items, gen_key: str | None,
                catalog: list[dict] | None = None) -> dict:
    """Summed per-stat deltas of the recognized entries in a mod/wishlist."""
    total = {s: 0 for s in BAR_STATS}
    for item in items or []:
        entry = _match_entry(item, catalog)
        if entry and gen_key:
            gen_deltas = entry["deltas"].get(gen_key, {})
            for s in BAR_STATS:
                total[s] += gen_deltas.get(s, 0)
    return total


def _compose_bars(car: dict, catalog: list[dict] | None = None) -> dict | None:
    """Deterministic stat bars: current = stock LLM baseline + summed deltas
    of recognized installed mods; dream = current + summed wishlist deltas.
    Both clamped 0-100. None until the stock baseline exists."""
    stats = car.get("stats")
    if not stats:
        return None
    gen = _gen_key(car.get("generation"))
    installed = _sum_deltas(car.get("mods"), gen, catalog)
    wished = _sum_deltas(car.get("wishlist"), gen, catalog)
    current, dream = {}, {}
    for s in BAR_STATS:
        current[s] = max(0, min(100, int(stats.get(s) or 0) + installed[s]))
        dream[s] = max(0, min(100, current[s] + wished[s]))
    return {"current": current, "dream": dream}


# --- Build fingerprints: what makes a portrait or stats block stale ---


def _identity_fp(car: dict) -> str:
    """Core identity (year/generation/trim). Mismatch => portrait regenerated."""
    return _car_desc(car)


def _build_fp(car: dict) -> str:
    """Visual build (color + mods). Mismatch => stored portrait gets EDITED."""
    return json.dumps({"color": str(car.get("color") or "").strip().lower(),
                       "mods": sorted(car.get("mods") or [])})


def _stats_fp(car: dict) -> str:
    """Identity only: stats are the STOCK baseline; installed mods compose
    deterministically via the catalog deltas, never via the LLM."""
    return json.dumps({"identity": _car_desc(car)})


def _portrait_action(stored: tuple[str | None, str | None] | None, car: dict) -> str:
    """'generate' | 'edit' | 'skip' for a car given the stored portrait's
    (identity_fp, build_fp), or None when no portrait exists yet."""
    if stored is None or stored[0] != _identity_fp(car):
        return "generate"
    if stored[1] != _build_fp(car):
        return "edit"
    return "skip"


STATS_PROMPT = """You are a Ford Mustang encyclopedia. For the {desc} Ford \
Mustang described below, reply with ONLY this JSON object, no other text:
{{"power": <0-100>, "acceleration": <0-100>, "top_speed": <0-100>, \
"handling": <0-100>, "braking": <0-100>, "hp": <stock horsepower, integer>, \
"zero_to_sixty": <stock 0-60 mph time in seconds, float>, \
"top_speed_mph": <stock top speed in mph, integer>}}
The 0-100 values are arcade-racing-game ratings calibrated across the entire \
Mustang range, 1964 to today: a 1974 Mustang II is roughly 15-25 power, a base \
1990s V6 around 30-40, a 2015+ GT around 75-85, a 2020 Shelby GT500 is 95-100.
The car is completely stock (factory condition, no modifications); rate it as \
such. Installed modifications are scored separately and must NOT be reflected \
here — every figure is the STOCK factory baseline."""


async def _generate_stats(car: dict) -> dict | None:
    """The car's STOCK baseline (mods compose on top via catalog deltas)."""
    desc = _car_desc(car)
    if not desc:
        return None
    resp = await _stats_llm.ainvoke(STATS_PROMPT.format(desc=desc))
    m = re.search(r"\{.*\}", resp.content, re.S)
    if not m:
        return None
    stats = json.loads(m.group())
    stats["fp"] = _stats_fp(car)  # cache key: recompute only when this drifts
    return stats


def _image_prompt(car: dict) -> str:
    color = car.get("color") or "a factory paint color appropriate to that generation"
    return (
        f"Photorealistic studio photograph, full side profile, of a {_car_desc(car)} "
        f"Ford Mustang in {color}. Body style exactly accurate for that model year "
        "and generation. Dark seamless studio background, soft reflective floor, "
        "dramatic rim lighting. No people, no text, no watermarks."
    )


async def _generate_image(user_id: str, car: dict) -> None:
    """From-scratch portrait (gpt-image-1): only for a car with no portrait yet
    or whose core identity changed. Stores both fingerprints alongside it."""
    prompt = _image_prompt(car)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{GATEWAY_URL}/images/generations",
            headers={"Authorization": f"Bearer {os.environ['AI_GATEWAY_API_KEY']}"},
            json={"model": "openai/gpt-image-1", "prompt": prompt,
                  "size": "1024x1024", "quality": "low"},
            timeout=300,
        )
    resp.raise_for_status()
    image = base64.b64decode(resp.json()["data"][0]["b64_json"])
    async with await _db() as conn:
        await conn.execute(
            "INSERT INTO car_images (user_id, car_id, image, prompt, identity_fp, build_fp) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (user_id, car_id) DO UPDATE "
            "SET image = EXCLUDED.image, prompt = EXCLUDED.prompt, "
            "identity_fp = EXCLUDED.identity_fp, build_fp = EXCLUDED.build_fp",
            (user_id, car["id"], image, prompt, _identity_fp(car), _build_fp(car)),
        )


def _edit_prompt(old_build_fp: str, car: dict) -> str:
    """Describe the DELTA between the stored portrait's build and the current
    one (mods added/removed, color change) as a photo-edit instruction."""
    old = json.loads(old_build_fp)
    mods = sorted(car.get("mods") or [])
    color = str(car.get("color") or "").strip().lower()
    parts = []
    color_changed = color != old.get("color", "")
    if color_changed:
        parts.append(f"repaint the car {car['color']}" if car.get("color")
                     else "repaint the car in a factory color for this model")
    added = [m for m in mods if m not in old.get("mods", [])]
    removed = [m for m in old.get("mods", []) if m not in mods]
    if added:
        parts.append("represent these newly installed modifications where visible: "
                     + ", ".join(added))
    if removed:
        parts.append("remove these modifications, restoring those areas to stock: "
                     + ", ".join(removed))
    keep = "angle, lighting, and background" if color_changed else \
        "color, angle, lighting, and background"
    return (
        "Edit this photo: " + "; ".join(parts) + ". If a modification is not "
        "externally visible, leave the photo unchanged in that respect. "
        f"Keep the car, {keep} otherwise identical."
    )


async def _edit_image(user_id: str, car: dict, old_build_fp: str, image: bytes) -> None:
    """Apply a build change as an EDIT to the stored portrait via
    gemini-2.5-flash-image. On any failure the old portrait is kept."""
    prompt = _edit_prompt(old_build_fp, car)
    b64 = base64.b64encode(image).decode()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{GATEWAY_URL}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['AI_GATEWAY_API_KEY']}"},
            json={"model": "google/gemini-2.5-flash-image", "messages": [{
                "role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]}]},
            timeout=300,
        )
    resp.raise_for_status()
    images = resp.json()["choices"][0]["message"].get("images") or []
    if not images:  # keep the old portrait; a later build change retries
        print(f"portrait edit returned no image for {user_id}/{car['id']}")
        return
    url = images[0]["image_url"]["url"]
    edited = base64.b64decode(url.split("base64,", 1)[1])
    async with await _db() as conn:
        await conn.execute(
            "UPDATE car_images SET image = %s, build_fp = %s "
            "WHERE user_id = %s AND car_id = %s",
            (edited, _build_fp(car), user_id, car["id"]),
        )


async def _sync_portrait(user_id: str, car: dict) -> None:
    """Bring the stored portrait up to date with the car: generate from scratch
    when missing or the identity changed, EDIT the stored photo when only the
    build (color/mods) changed, otherwise leave the bytes untouched."""
    async with await _db() as conn:
        row = await (
            await conn.execute(
                "SELECT prompt, identity_fp, build_fp, image FROM car_images "
                "WHERE user_id = %s AND car_id = %s",
                (user_id, car["id"]),
            )
        ).fetchone()
    if row and row[1] is None:  # pre-fingerprint row: adopt it, never re-roll
        if row[0] == _image_prompt(car):
            async with await _db() as conn:
                await conn.execute(
                    "UPDATE car_images SET identity_fp = %s, build_fp = %s "
                    "WHERE user_id = %s AND car_id = %s",
                    (_identity_fp(car), _build_fp(car), user_id, car["id"]),
                )
            return
        row = None  # identity/color drifted while unfingerprinted -> regenerate
    action = _portrait_action((row[1], row[2]) if row else None, car)
    if action == "generate":
        await _generate_image(user_id, car)
    elif action == "edit":
        await _edit_image(user_id, car, row[2], bytes(row[3]))


async def _save_car_stats(user_id: str, car_id: str, stats: dict) -> None:
    """Re-read-modify-write so a concurrent chat turn's garage write between
    enrichment start and finish isn't clobbered; the stats only land if the
    build they were computed for is still the car's current build."""
    async with await _db() as conn:
        profile = await _load_profile(conn, user_id)
        for c in profile.get("cars", []):
            if c["id"] == car_id and _stats_fp(c) == stats.get("fp"):
                c["stats"] = stats
                await conn.execute(
                    "UPDATE garage SET profile = %s WHERE user_id = %s",
                    (Json(profile), user_id),
                )
                return


_enriching: set[str] = set()  # ponytail: in-process dedupe; a queue if multi-worker


async def _enrich_garage(user_id: str) -> None:
    """Background: fill in missing stats and missing/stale portraits for every
    identified car. Cheap no-op when everything is already current."""
    if user_id in _enriching:
        return
    _enriching.add(user_id)
    try:
        async with await _db() as conn:
            profile = await _load_profile(conn, user_id)
        for car in profile.get("cars", []):
            if not _car_desc(car):
                continue  # nothing identifying to draw or rate yet
            cached = car.get("stats")
            if not cached or cached.get("fp") != _stats_fp(car):
                stats = await _generate_stats(car)
                if stats:
                    await _save_car_stats(user_id, car["id"], stats)
            await _sync_portrait(user_id, car)
    except Exception as e:  # ponytail: enrichment is best-effort, next request retries
        print(f"garage enrichment failed for {user_id}: {e}")
    finally:
        _enriching.discard(user_id)


@tool
async def update_garage(
    config: RunnableConfig,
    car: str | None = None,
    year: int | None = None,
    trim: str | None = None,
    generation: str | None = None,
    color: str | None = None,
    nickname: str | None = None,
    mods: list[str] | None = None,
    wishlist: list[str] | None = None,
    goals: list[str] | None = None,
) -> str:
    """Record facts the user reveals about their own Mustang(s). Call this
    IMMEDIATELY the FIRST time the user mentions a car of theirs, with
    whatever facts are known at that moment — partial info is expected and
    fine: e.g. the user says "I have a blue Mustang GT" → call
    update_garage(car="blue Mustang GT", trim="GT", color="blue") right away,
    no year needed. NEVER wait for complete details; update the same car as
    more arrives in later turns. The garage holds multiple cars. Pass `car`
    with the user's own words for which car they mean (e.g. "my 2016 GT",
    "the Fox-body") whenever they own more than one or mention a car not yet
    in their garage — an unknown car automatically becomes a new garage
    entry. Per-car facts: model year, trim (e.g. GT Premium), generation
    (e.g. S550), color, nickname, installed mods, and wishlist/planned mods.
    goals (track, show, daily driver) apply to the user as a whole. All
    arguments optional; new values merge into the existing profile."""
    user_id = config["configurable"]["user_id"]
    updates = {
        k: v
        for k, v in dict(year=year, trim=trim, generation=generation,
                         color=color, nickname=nickname,
                         mods=mods, wishlist=wishlist).items()
        if v is not None
    }
    async with await _db() as conn:
        # ponytail: read-merge-write, no row lock; fine for one-user-per-thread chat
        profile = await _load_profile(conn, user_id)
        cars = profile.setdefault("cars", [])
        if updates or car:
            target = _match_car(cars, car, updates)
            if target is None:
                target = {"id": uuid.uuid4().hex[:8]}
                cars.append(target)
            for k, v in updates.items():
                if isinstance(v, list):
                    current = target.get(k, [])
                    target[k] = current + [x for x in v if x not in current]
                else:
                    target[k] = v
            _autofill_generation(target)
            if target.get("stats") and target["stats"].get("fp") != _stats_fp(target):
                target["stats"] = None  # identity changed -> recompute stock baseline
        if goals:
            current = profile.get("goals", [])
            profile["goals"] = current + [g for g in goals if g not in current]
        await conn.execute(
            "INSERT INTO garage (user_id, profile) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET profile = EXCLUDED.profile",
            (user_id, Json(profile)),
        )
    return "Garage profile updated."


@tool
async def update_instructions(instruction: str, config: RunnableConfig) -> str:
    """Record a standing preference about how the user wants answers (e.g.
    "keep answers short", "explain like I'm a beginner"). Applied to every
    future answer."""
    user_id = config["configurable"]["user_id"]
    async with await _db() as conn:
        await conn.execute(
            "INSERT INTO instructions (user_id, instruction) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (user_id, instruction),
        )
    return "Instruction saved."


def _prompt(state, config: RunnableConfig):
    """Per-turn system prompt: base persona + garage profile + instructions,
    assembled in /chat and passed through config."""
    system = config["configurable"].get("system_prompt", SYSTEM_PROMPT)
    return [{"role": "system", "content": system}] + state["messages"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qdrant, _agent
    _qdrant = build_index()
    async with AsyncPostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as saver:
        await saver.setup()  # idempotent
        async with await _db() as conn:  # idempotent memory tables
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS garage ("
                "user_id TEXT PRIMARY KEY, profile JSONB NOT NULL DEFAULT '{}')"
            )
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS instructions ("
                "id BIGSERIAL PRIMARY KEY, user_id TEXT NOT NULL, "
                "instruction TEXT NOT NULL, UNIQUE (user_id, instruction))"
            )
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS summaries ("
                "user_id TEXT NOT NULL, session_id TEXT NOT NULL, "
                "summary TEXT NOT NULL, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                "PRIMARY KEY (user_id, session_id))"
            )
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS car_images ("
                "user_id TEXT NOT NULL, car_id TEXT NOT NULL, "
                "image BYTEA NOT NULL, prompt TEXT NOT NULL, "
                "identity_fp TEXT, build_fp TEXT, "
                "PRIMARY KEY (user_id, car_id))"
            )
            await conn.execute(  # pre-2.1 installs lack the fingerprint columns
                "ALTER TABLE car_images ADD COLUMN IF NOT EXISTS identity_fp TEXT"
            )
            await conn.execute(
                "ALTER TABLE car_images ADD COLUMN IF NOT EXISTS build_fp TEXT"
            )
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS identities ("
                "google_sub TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
                "email TEXT, name TEXT, picture TEXT, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS chats ("
                "user_id TEXT NOT NULL, chat_id TEXT NOT NULL, "
                "title TEXT NOT NULL, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                "PRIMARY KEY (user_id, chat_id))"
            )
        _agent = create_react_agent(
            ChatOpenAI(
                model="anthropic/claude-sonnet-4.5",
                base_url=GATEWAY_URL,
                api_key=os.environ["AI_GATEWAY_API_KEY"],
            ),
            [search_archive, web_search, check_recalls, recommend_products,
             update_garage, update_instructions],
            prompt=_prompt,
            checkpointer=saver,
        )
        yield


app = FastAPI(lifespan=lifespan)
# ponytail: open CORS; identity comes from the bearer token, not the origin
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --- Optional Google login (issue #17): identity resolution, not data migration.
# identities maps google_sub -> canonical internal user_id; everything else in
# the system stays keyed by user_id. First-ever login adopts the browser's
# anonymous UUID in place; later logins (any device) return the canonical id.


def _verify_google_token(id_token: str) -> dict:
    """Verify a Google ID token against Google's JWKS; returns its claims.
    The seam tests monkeypatch — everything past here runs for real."""
    return google_id_token.verify_oauth2_token(
        id_token, google_requests.Request(), os.environ["GOOGLE_CLIENT_ID"]
    )


class GoogleLogin(BaseModel):
    id_token: str
    anon_user_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)
    ]


@app.post("/auth/google")
async def auth_google(body: GoogleLogin):
    """Exchange a Google ID token for an app JWT + the canonical user_id.
    Unknown google_sub: bind it to the caller's anonymous user_id (their
    existing garage/chats are adopted in place, zero data movement)."""
    if not os.environ.get("GOOGLE_CLIENT_ID") or not os.environ.get("AUTH_JWT_SECRET"):
        raise HTTPException(503, "Google sign-in is not configured")
    try:
        claims = _verify_google_token(body.id_token)
    except Exception:  # bad signature, expired, wrong audience/issuer
        raise HTTPException(401, "invalid Google token")
    async with await _db() as conn:
        row = await (
            await conn.execute(
                # existing sub keeps its user_id (canonical); profile fields refresh
                "INSERT INTO identities (google_sub, user_id, email, name, picture) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (google_sub) DO UPDATE SET "
                "email = EXCLUDED.email, name = EXCLUDED.name, "
                "picture = EXCLUDED.picture RETURNING user_id",
                (claims["sub"], body.anon_user_id, claims.get("email"),
                 claims.get("name"), claims.get("picture")),
            )
        ).fetchone()
    user_id = row[0]
    token = jwt.encode(
        {"sub": claims["sub"], "uid": user_id,
         "exp": datetime.now(timezone.utc) + timedelta(days=30)},
        os.environ["AUTH_JWT_SECRET"], algorithm="HS256",
    )
    return {"token": token, "user_id": user_id, "name": claims.get("name"),
            "email": claims.get("email"), "picture": claims.get("picture")}


async def _auth_uid(authorization: Annotated[str | None, Header()] = None) -> str | None:
    """The canonical user_id from a valid app-JWT bearer; None when anonymous.
    Invalid/expired tokens 401 explicitly so the client clears its token and
    retries anonymously, rather than silently acting as the wrong user."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        return jwt.decode(authorization[7:], os.environ.get("AUTH_JWT_SECRET", ""),
                          algorithms=["HS256"])["uid"]
    except jwt.InvalidTokenError:
        raise HTTPException(401, "invalid or expired token")


AuthUid = Annotated[str | None, Depends(_auth_uid)]


@app.get("/auth/me")
async def auth_me(auth_uid: AuthUid = None):
    """Restore session state from a stored app JWT."""
    if not auth_uid:
        raise HTTPException(401, "not signed in")
    async with await _db() as conn:
        row = await (
            await conn.execute(
                "SELECT name, email, picture FROM identities WHERE user_id = %s LIMIT 1",
                (auth_uid,),
            )
        ).fetchone()
    if not row:
        raise HTTPException(401, "unknown identity")
    return {"user_id": auth_uid, "name": row[0], "email": row[1], "picture": row[2]}


class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: str = "default"  # old clients without one all share this bucket
    chat_id: str = "default"  # the pre-multi-chat thread survives as "default"


def _thread_id(user_id: str, chat_id: str) -> str:
    """LangGraph thread per chat; "default" keeps the legacy bare-user_id
    thread so pre-multi-chat history survives."""
    return user_id if chat_id == "default" else f"{user_id}:{chat_id}"


def _msg_text(msg) -> str:
    """Plain text of a LangChain message; content blocks flattened."""
    c = msg.content
    if isinstance(c, list):
        c = "".join(b.get("text", "") for b in c
                    if isinstance(b, dict) and b.get("type") == "text")
    return c if isinstance(c, str) else ""


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/garage/{user_id}")
async def garage(user_id: str, auth_uid: AuthUid = None):
    user_id = auth_uid or user_id  # signed-in identity wins over the path id
    profile, instructions, summaries = await _get_memory(user_id)
    if profile.get("cars"):
        # opportunistic fire-and-forget: fills stats/portraits missed earlier;
        # _enrich_garage no-ops cheaply when everything is current
        asyncio.get_running_loop().create_task(_enrich_garage(user_id))
        # composed current/dream bars ride along so clients never re-derive them
        profile = {**profile,
                   "cars": [{**c, "bars": _compose_bars(c)} for c in profile["cars"]]}
    return {
        "profile": profile,
        "instructions": instructions,
        "summaries": [{"summary": s["summary"], "date": s["date"]} for s in summaries],
    }


@app.get("/garage/{user_id}/cars/{car_id}/image")
async def car_image(user_id: str, car_id: str, auth_uid: AuthUid = None):
    user_id = auth_uid or user_id
    async with await _db() as conn:
        row = await (
            await conn.execute(
                "SELECT image FROM car_images WHERE user_id = %s AND car_id = %s",
                (user_id, car_id),
            )
        ).fetchone()
    if not row:
        raise HTTPException(404, "portrait not generated yet")
    return Response(
        content=bytes(row[0]),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/garage/{user_id}/cars/{car_id}/shop")
async def upgrade_shop(user_id: str, car_id: str, auth_uid: AuthUid = None):
    """The car's Upgrade Shop: a recommended strip of 2-3 eligible sponsor
    products (category fit against this car's generation, current mods, and
    wishlist gaps) plus the full catalog — sponsor products and generic mod
    categories — each row carrying the five delta chips for this car's
    generation. The have-it/want-it actions write through the existing
    car PATCH endpoint; this route only reads."""
    user_id = auth_uid or user_id
    async with await _db() as conn:
        profile = await _load_profile(conn, user_id)
    car = next((c for c in profile.get("cars", []) if c["id"] == car_id), None)
    if car is None:
        raise HTTPException(404, "car not found")
    gen = _gen_key(car.get("generation"))
    owned = [e for e in (_match_entry(m) for m in car.get("mods") or []) if e]
    wished = [e for e in (_match_entry(w) for w in car.get("wishlist") or []) if e]
    owned_ids = {e["id"] for e in owned}
    wished_ids = {e["id"] for e in wished}

    def row(entry: dict) -> dict:
        return {
            "id": entry["id"], "name": entry["name"],
            "advertiser": entry["advertiser"], "sponsored": entry["sponsored"],
            "description": entry["description"], "categories": entry["categories"],
            "image": entry["image"], "link": entry["link"],
            "deltas": entry["deltas"].get(gen) if gen else None,
            "installed": entry["id"] in owned_ids,
            "wishlisted": entry["id"] in wished_ids,
        }

    # The browsable catalog: recommendable sponsor products + generic mod
    # categories. Non-recommendable ad campaigns are not upgrades.
    rows = [row(e) for e in CATALOG if e["recommendable"] or not e["sponsored"]]
    candidates = [r for r in rows
                  if r["sponsored"] and not r["installed"] and not r["wishlisted"]]
    # wishlist gaps: categories the build or the wishlist already fills
    covered = {c for e in owned + wished for c in e.get("categories", [])}
    fit = [r for r in candidates if not (set(r["categories"]) & covered)] or candidates

    def gain(r: dict) -> int:
        return sum(r["deltas"].values()) if r["deltas"] else 0

    return {"recommended": sorted(fit, key=gain, reverse=True)[:3], "catalog": rows}


Str80 = Annotated[str, StringConstraints(strip_whitespace=True, max_length=80)]
Item = Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)]


class CarPatch(BaseModel):
    """UI edits; None clears a scalar field, lists replace wholesale."""
    year: int | None = Field(None, ge=1964, le=2027)
    trim: Str80 | None = None
    generation: Str80 | None = None
    color: Str80 | None = None
    nickname: Str80 | None = None
    mods: list[Item] | None = Field(None, max_length=50)
    wishlist: list[Item] | None = Field(None, max_length=50)


ReqStr80 = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)
]


class CarCreate(BaseModel):
    """Standardized picker intake: the three required identity fields."""
    year: int = Field(ge=1964, le=2027)
    trim: ReqStr80
    color: ReqStr80
    nickname: ReqStr80 | None = None


@app.post("/garage/{user_id}/cars", status_code=201)
async def create_car(user_id: str, body: CarCreate, background_tasks: BackgroundTasks,
                     auth_uid: AuthUid = None):
    """Create a car from the picker; generation derives from the year and
    enrichment (stats + portrait) runs in the background, like PATCH."""
    user_id = auth_uid or user_id
    car = {"id": uuid.uuid4().hex[:8], **body.model_dump(exclude_none=True)}
    _autofill_generation(car)
    async with await _db() as conn:
        profile = await _load_profile(conn, user_id)
        cars = profile.setdefault("cars", [])
        # ponytail: same year+trim = same car -> 409; merging risks clobbering
        if any(str(c.get("year")) == str(car["year"])
               and str(c.get("trim", "")).lower() == car["trim"].lower()
               for c in cars):
            raise HTTPException(409, "that year and trim is already in the garage")
        cars.append(car)
        await conn.execute(
            "INSERT INTO garage (user_id, profile) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET profile = EXCLUDED.profile",
            (user_id, Json(profile)),
        )
    background_tasks.add_task(_enrich_garage, user_id)
    return {**car, "bars": _compose_bars(car)}


@app.patch("/garage/{user_id}/cars/{car_id}")
async def patch_car(user_id: str, car_id: str, patch: CarPatch,
                    background_tasks: BackgroundTasks, auth_uid: AuthUid = None):
    user_id = auth_uid or user_id
    updates = patch.model_dump(exclude_unset=True)
    async with await _db() as conn:
        profile = await _load_profile(conn, user_id)
        target = next((c for c in profile.get("cars", []) if c["id"] == car_id), None)
        if target is None:
            raise HTTPException(404, "car not found")
        for k, v in updates.items():
            if v is None:
                target.pop(k, None)
            else:
                target[k] = v
        _autofill_generation(target)
        if target.get("stats") and target["stats"].get("fp") != _stats_fp(target):
            target["stats"] = None  # identity changed -> recompute stock baseline
        await conn.execute(
            "UPDATE garage SET profile = %s WHERE user_id = %s",
            (Json(profile), user_id),
        )
    background_tasks.add_task(_enrich_garage, user_id)
    return {**target, "bars": _compose_bars(target)}


@app.delete("/garage/{user_id}/cars/{car_id}", status_code=204)
async def delete_car(user_id: str, car_id: str, auth_uid: AuthUid = None):
    user_id = auth_uid or user_id
    async with await _db() as conn:
        profile = await _load_profile(conn, user_id)
        cars = profile.get("cars", [])
        if not any(c["id"] == car_id for c in cars):
            raise HTTPException(404, "car not found")
        profile["cars"] = [c for c in cars if c["id"] != car_id]
        await conn.execute(
            "UPDATE garage SET profile = %s WHERE user_id = %s",
            (Json(profile), user_id),
        )
        await conn.execute(
            "DELETE FROM car_images WHERE user_id = %s AND car_id = %s",
            (user_id, car_id),
        )


@app.get("/chats/{user_id}")
async def list_chats(user_id: str, auth_uid: AuthUid = None):
    """The user's chats, recent-first. A legacy bare-user_id thread that has
    checkpointer state but no chats row yet gets its row backfilled here."""
    user_id = auth_uid or user_id
    async with await _db() as conn:
        rows = await (
            await conn.execute(
                "SELECT chat_id, title, updated_at FROM chats "
                "WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,),
            )
        ).fetchall()
    chats = [{"chat_id": r[0], "title": r[1], "updated_at": r[2].isoformat()}
             for r in rows]
    if not any(c["chat_id"] == "default" for c in chats):
        snap = await _agent.aget_state({"configurable": {"thread_id": user_id}})
        msgs = (snap.values or {}).get("messages") if snap else None
        if msgs:
            title = next((_msg_text(m)[:60] for m in msgs
                          if m.type == "human" and _msg_text(m)), "Earlier conversation")
            async with await _db() as conn:
                row = await (
                    await conn.execute(
                        "INSERT INTO chats (user_id, chat_id, title) "
                        "VALUES (%s, 'default', %s) "
                        "ON CONFLICT (user_id, chat_id) DO UPDATE SET title = chats.title "
                        "RETURNING updated_at",
                        (user_id, title),
                    )
                ).fetchone()
            chats.append({"chat_id": "default", "title": title,
                          "updated_at": row[0].isoformat()})
    return chats


@app.get("/chats/{user_id}/{chat_id}/messages")
async def chat_messages(user_id: str, chat_id: str, auth_uid: AuthUid = None):
    """The chat's transcript, reconstructed from the checkpointer: user and
    assistant turns only (tool calls/results and empty AI messages skipped)."""
    user_id = auth_uid or user_id
    snap = await _agent.aget_state(
        {"configurable": {"thread_id": _thread_id(user_id, chat_id)}}
    )
    out = []
    for m in ((snap.values or {}).get("messages", []) if snap else []):
        text = _msg_text(m)
        if not text:
            continue
        if m.type == "human":
            out.append({"role": "user", "content": text})
        elif m.type == "ai":
            out.append({"role": "assistant", "content": text})
    return out


async def _post_turn(user_id: str, session_id: str, user_text: str, parts: list[str]):
    """After the SSE stream: update the session summary, then fill in any
    missing car stats/portraits the turn's garage writes created."""
    await _summarize(user_id, session_id, user_text, parts)
    await _enrich_garage(user_id)


@app.post("/chat")
async def chat(req: ChatRequest, auth_uid: AuthUid = None):
    user_id = auth_uid or req.user_id  # signed-in identity wins over the body id
    profile, instructions, summaries = await _get_memory(user_id)
    async with await _db() as conn:  # title = first user message, then just bump
        await conn.execute(
            "INSERT INTO chats (user_id, chat_id, title) VALUES (%s, %s, %s) "
            "ON CONFLICT (user_id, chat_id) DO UPDATE SET updated_at = now()",
            (user_id, req.chat_id, req.message[:60]),
        )
    system = SYSTEM_PROMPT
    if profile:
        lean = {**profile, "cars": [  # stats are UI data, not prompt data
            {k: v for k, v in c.items() if k != "stats"} for c in profile.get("cars", [])
        ]} if profile.get("cars") else profile
        system += f"\n\nThe user's garage profile (their cars):\n{json.dumps(lean)}"
    if instructions:
        system += "\n\nStanding user instructions (follow in every answer):\n" + "\n".join(
            f"- {i}" for i in instructions
        )
    past = [s for s in summaries if s["session_id"] != req.session_id][:5]
    if past:
        system += "\n\nPrevious conversations with this user (most recent first):\n" + "\n".join(
            f"- [{s['date']}] {s['summary']}" for s in past
        )

    answer_parts: list[str] = []  # summarizer input, filled as tokens stream

    async def sse():
        citations, seen = [], set()
        ads_sent = 0  # hard cap regardless of how often the tool fires
        # ponytail: "active car" = first garage car; pass a car_id in
        # ChatRequest if multi-car users ever need per-chat targeting
        active_gen = _gen_key((profile.get("cars") or [{}])[0].get("generation"))
        stream = _agent.astream(
            {"messages": [{"role": "user", "content": req.message}]},
            {"configurable": {
                "thread_id": _thread_id(user_id, req.chat_id),
                "user_id": user_id,
                "system_prompt": system,
            }},
            stream_mode="messages",
        )
        async for msg, _meta in stream:
            if isinstance(msg, ToolMessage):
                yield f"data: {json.dumps({'type': 'tool', 'name': msg.name})}\n\n"
                if msg.name == "search_archive":
                    for hit in json.loads(msg.content):
                        if hit["url"] not in seen:
                            seen.add(hit["url"])
                            citations.append({"title": hit["title"], "url": hit["url"]})
                elif msg.name == "recommend_products":
                    for item in json.loads(msg.content):
                        entry = _RECO_BY_ID.get(item["id"])
                        if not entry or ads_sent >= AD_TOP_K:
                            continue
                        ads_sent += 1
                        ad = {
                            "type": "ad",
                            "product": entry["name"],
                            "advertiser": entry["advertiser"],
                            "description": entry["description"],
                            "image": entry["image"],
                            "link": entry["link"],
                            "sponsored": True,
                            "deltas": entry["deltas"].get(active_gen)
                            if active_gen else None,
                        }
                        yield f"data: {json.dumps(ad)}\n\n"
            elif isinstance(msg, AIMessageChunk) and isinstance(msg.content, str) and msg.content:
                answer_parts.append(msg.content)
                yield f"data: {json.dumps({'type': 'token', 'text': msg.content})}\n\n"
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        background=BackgroundTask(
            _post_turn, user_id, req.session_id, req.message, answer_parts
        ),
    )
