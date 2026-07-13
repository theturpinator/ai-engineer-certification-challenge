"""Ask MustangDriver chat API.

FastAPI app: POST /chat streams SSE tokens + tool_start/tool/citations/ad
events, ~10s keepalive pings during silence, and a generic error event on
stream failure (real exception server-log only), from a
LangGraph ReAct agent with four tools: search_archive (in-memory Qdrant over
index_artifact/, built at startup), web_search (Tavily, live web),
check_recalls (NHTSA Recalls API), and recommend_products (intent-gated
sponsored recommendations over the committed ads_artifact/ catalog — product
entries plus advertiser-level entries whose card links the sponsor's
canonical website — via a mini BM25+dense RRF index; at most three `ad`
events per turn, each carrying
the creative image, UTM click-through link, Sponsored flag, and stat-delta
chips for the user's car), plus two memory tools: update_garage (semantic:
the user's cars, per-car mods/wishlist with add, remove, and move
semantics, user-level goals) and
update_instructions (procedural: standing answer preferences), both keyed by
user_id in Postgres and folded into the system prompt each turn. Episodic
memory: after each turn a background task has Claude Haiku keep a rolling 2-3
sentence summary of the session (keyed by user_id + session_id from the
client), and recent past-session summaries are injected into the system
prompt. The garage holds multiple cars (profile.cars, legacy flat profiles
migrate on read); each car is enriched in the background with arcade-style
STOCK-baseline stats (Sonnet, critically calibrated, the safety score
grounded in the car's public NHTSA 5-Star overall rating when one exists —
carried in stats.nhtsa for the UI to link — validated before caching, and
cached in the car with an identity fingerprint) and a one-time AI-generated
photorealistic portrait (gpt-image-1 via the gateway, cached in car_images). The
portrait is seeded exactly once, at car creation, then frozen forever: no
identity, color, or mod change ever spends another image-model call (issue
#34). PUT /garage/{user_id}/cars/{car_id}/image (raw image bytes, <=8 MB)
replaces the portrait with the user's own photo, which is canonical and
never overwritten. Garages are capped at 10 cars on BOTH creation paths
(POST /garage/{user_id}/cars returns 400 "Garage is full — max 10 cars"; the
chat update_garage tool returns the same message for the agent to relay);
over-cap garages are grandfathered (issue #35). Bars compose deterministically from the committed catalog:
current = stock + summed deltas of recognized installed mods, dream =
current + wishlist deltas, both clamped 0-100 and returned by the garage
read/write endpoints so clients never re-derive them. GET
/garage/{user_id}/cars/{car_id}/shop serves the Upgrade Shop (recommended
sponsor products + the full searchable catalog with per-car delta chips);
its have-it/want-it actions write through the existing car PATCH. GET
/garage/{user_id} returns everything known (each car carries composed bars
plus photo_uploaded, which drives the upload-pill UI); cars are editable via
PATCH/DELETE /garage/{user_id}/cars/{car_id}. First-run onboarding (issue
#46) runs inside the chat: a brand-new user's client POSTs the sentinel
message "[begin onboarding]", which stamps profile.onboarded=false and
injects an interview script into the system prompt — the agent asks the
profile questions one at a time (name, their Mustang, goals, and how they
like answers formatted), records them via update_garage and
update_instructions, and finishes by calling the complete_onboarding tool
(profile.onboarded=true, the client's unlock signal). The sentinel never appears in transcript
replays or chat titles. Users can hold multiple chats:
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
import traceback
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
from fastapi import (BackgroundTasks, Depends, FastAPI, Header, HTTPException,
                     Request)
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
- web_search when the archive comes up empty or doesn't actually answer \
what was asked (wrong model year, wrong generation, missing the specifics), \
or when the question is inherently live (current prices, market values, \
news, upcoming events, availability). For inherently live questions the \
answer MUST begin with "According to a live web search" and cite the source \
pages. When you fall back to the web because the archive lacked the answer, \
just answer from the web results with citations — NEVER tell the user the \
archive was empty, limited, or off-target, and never mention falling back.
- Never narrate your search process ("Let me search…", "The archive results \
focus on…"). Call tools silently and give only the answer.
- check_recalls for safety-recall questions; report the official NHTSA \
campaigns (component, summary, remedy, date), or that none were found.
- recommend_products whenever the user is shopping for a part or upgrade, \
asks where to buy something, asks for upgrade advice, or the answer \
naturally calls for a specific product — ALWAYS call it before answering \
such questions, never from memory alone. Call it at most once per turn. \
Weave the fitting product(s) into your answer, naming the advertiser behind \
each; on where-to-buy questions recommend the fitting sponsor's site when \
the results include one. NEVER call it on history, \
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
calling update_garage with its facts. When the user says a mod or wishlist \
item is gone ("I took the intake off", "sold the supercharger", "drop the \
exhaust from my wishlist"), call update_garage with remove_mods / \
remove_wishlist; "I installed the X from my wishlist" is remove_wishlist \
plus mods in one call, and "replaced X with Y" is remove_mods=[X] plus \
mods=[Y]. NEVER tell the user a removal happened unless the tool call \
succeeded — if the tool reports the item wasn't found, say so. Never \
announce or mention that you recorded anything you were not asked about.
- Whenever the user states a standing preference about how you should answer \
(e.g. "keep answers short", "always end with X"), silently call \
update_instructions with that preference, then follow it.
- Always follow every standing instruction listed below, in every answer."""

# First-run onboarding (issue #46): the client opens a brand-new user's chat
# with this exact hidden message; it never shows in transcripts or titles.
ONBOARD_KICKOFF = "[begin onboarding]"

ONBOARDING_PROMPT = """

FIRST-RUN ONBOARDING IS IN PROGRESS. This is a brand-new user; before \
normal Q&A, interview them to build their profile. Greet them warmly and \
briefly, then ask these questions ONE AT A TIME — exactly one question per \
message, always waiting for the user's reply before the next:
1. What can I call you? (record with update_instructions, e.g. "Address \
the user as Jake", and use their name from then on)
2. Do you have a Mustang? If yes, ask what it is (year, trim, color, and a \
nickname if it has one) and record it with update_garage. Partial answers \
are fine — ask once for anything missing, then move on.
3. Are you planning any upgrades or mods? (yes → update_garage goal \
"Planning upgrades")
4. Do you think you'll buy another Mustang some day? (yes → goal \
"Shopping for a future Mustang")
5. Want to keep up with car shows and events? (yes → goal "Interested in \
car shows and events")
6. Are you into track days — or want to be? (yes → goal "Track days")
7. Last one: how do you like your answers — short and to the point or \
detailed, bullet lists or prose? (record with update_instructions, e.g. \
"Keep answers short, in bullet lists")
Record every answer silently the moment it arrives — car facts and goals \
via update_garage, name and answer style via update_instructions, each \
BEFORE you reply, including the final question's. If one reply answers \
several questions, record them all and skip ahead. If the user wants to \
skip onboarding or clearly won't play along, stop asking. After the final \
answer is recorded (or on skip), call complete_onboarding, then send a \
short thank-you: their profile is saved and they can now ask anything \
Mustang. Do not answer unrelated questions until onboarding is complete — \
give a one-line answer at most, then return to the current question. The \
user's message "[begin onboarding]" is an automatic trigger, not something \
they typed — never mention or quote it."""

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

# Hard ceiling on per-user portrait-seeding spend (issue #35). Enforced on
# BOTH creation paths (picker endpoint and the chat tool, which appends to the
# profile directly); users already over the cap are grandfathered — the check
# only blocks NEW cars.
MAX_CARS = 10
GARAGE_FULL_MSG = "Garage is full — max 10 cars"


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


def _dedupe(items: list) -> list:
    """Order-preserving, case-insensitive dedupe for mods/wishlist: a retried
    add or a stale-state PATCH must not store an item twice (issue #29)."""
    seen: set[str] = set()
    out = []
    for x in items:
        k = str(x).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out


async def _load_profile(conn, user_id: str) -> dict:
    """Garage profile, migrated to the cars[] shape. A legacy flat profile is
    written back once so the migrated car id is stable across reads; ditto
    duplicated mods/wishlist entries, so old data self-heals (issue #29)."""
    row = await (
        await conn.execute("SELECT profile FROM garage WHERE user_id = %s", (user_id,))
    ).fetchone()
    raw = row[0] if row else {}
    profile = _migrate(dict(raw))
    changed = profile != raw
    for car in profile.get("cars", []):
        for k in ("mods", "wishlist"):
            deduped = _dedupe(car.get(k) or [])
            if deduped != (car.get(k) or []):
                car[k] = deduped
                changed = True
    if changed:
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
AD_TOP_K = 3  # at most three sponsored cards per turn (issue #50)
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


def _list_remove(current: list[str], phrases: list[str]) -> tuple[list[str], list[str]]:
    """(kept, unmatched_phrases): drop items matching any phrase by normalized
    whole-word containment in either direction, so "the cold air intake"
    removes "Cold air intake" and vice versa."""
    def hit(item: str, phrase: str) -> bool:
        a, b = f" {_norm_words(item)} ", f" {_norm_words(phrase)} "
        return a in b or b in a
    kept = [x for x in current if not any(hit(x, p) for p in phrases)]
    missed = [p for p in phrases if not any(hit(x, p) for x in current)]
    return kept, missed


def _apply_car_updates(target: dict, updates: dict, removals: dict) -> list[str]:
    """Mutate target: removals first, then the additive merge, so a swap in
    one call ("replaced X with Y") lands as remove+add. Returns removal
    phrases that matched nothing (the agent tells the user rather than
    silently confirming)."""
    missed: list[str] = []
    for k, phrases in removals.items():
        phrases = [p for p in phrases if _norm_words(p)]  # "" would match everything
        if phrases:
            target[k], miss = _list_remove(target.get(k, []), phrases)
            missed += miss
    for k, v in updates.items():
        if isinstance(v, list):
            current = target.get(k, [])
            target[k] = current + [x for x in v if x not in current]
        else:
            target[k] = v
    return missed


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


# --- Staleness: stats recompute on identity drift; portraits never do ---


def _stats_fp(car: dict) -> str:
    """Identity only: stats are the STOCK baseline; installed mods compose
    deterministically via the catalog deltas, never via the LLM. The schema
    version regenerates every cached baseline when the stat set changes
    (v2 = ownership stats, issue #27; v3 = critical calibration + NHTSA
    grounding, issue #32 — also flushes cached all-zero baselines, #28)."""
    return json.dumps({"identity": _car_desc(car), "v": 3})


def _portrait_action(stored) -> str:
    """'generate' | 'skip' for a car given its stored portrait row (None when
    no portrait exists yet). The portrait is seeded exactly once — when no row
    exists — then frozen forever: ANY existing row (generated or user-uploaded,
    fingerprints current or drifted) is a no-op, and there is no edit verdict.
    Image spend is thus bounded at one generation per car (issue #34)."""
    return "skip" if stored is not None else "generate"


STATS_PROMPT = """You are a critical automotive reviewer, not a salesman. For \
the {desc} Ford Mustang described below, reply with ONLY this JSON object, no \
other text:
{{"power": <0-100>, "acceleration": <0-100>, "top_speed": <0-100>, \
"handling": <0-100>, "braking": <0-100>, "style": <0-100>, \
"comfort": <0-100>, "safety": <0-100>, "reliability": <0-100>, \
"hp": <stock horsepower, integer>, \
"zero_to_sixty": <stock 0-60 mph time in seconds, float>, \
"top_speed_mph": <stock top speed in mph, integer>}}
The 0-100 values are arcade-racing-game ratings calibrated against ALL cars \
on the road today, not just Mustangs: 50 is an average modern car, and 90+ is \
reserved for the absolute best in class of anything on sale. Be critical, not \
flattering — most Mustangs land mid-pack in most categories.
Performance (power, acceleration, top_speed, handling, braking): a 1974 \
Mustang II is roughly 15-25 power, a base 1990s V6 around 30-40, a 2015+ GT \
around 70-80; a 2020 Shelby GT500 nears 95 on power, but no Mustang \
out-handles a purpose-built sports car.
Ownership (style, comfort, safety, reliability): judge the stock car \
honestly. Safety: {grounding} A two-door sports coupe is never a 90+ safety \
car — a modern 5-star-rated Mustang belongs around 65-80, a 4-star car \
50-65, and pre-airbag classics 5-20. Comfort: firm sporty coupes rarely \
clear 70. Reliability: reflect the platform's real reputation, known \
problem years, and its age.
The car is completely stock (factory condition, no modifications); rate it as \
such. Installed modifications are scored separately and must NOT be reflected \
here — every figure is the STOCK factory baseline. If this exact year/trim \
combination was never actually produced, rate the closest real Mustang \
matching the description — never refuse and never return zeros."""


async def _nhtsa_rating(year) -> dict | None:
    """NHTSA 5-Star overall rating for the Mustang model year (public API, no
    key — same agency as check_recalls). None when unrated (pre-2011 model
    years) or unreachable: grounding is best-effort by design."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"https://api.nhtsa.gov/SafetyRatings/modelyear/{int(year)}"
                "/make/FORD/model/MUSTANG")
            results = r.json().get("Results") or []
            if not results:
                return None
            # prefer the coupe ("2 DR") variant over the convertible ("C")
            vid = next((v for v in results if "2 DR" in v["VehicleDescription"]),
                       results[0])["VehicleId"]
            r = await client.get(
                f"https://api.nhtsa.gov/SafetyRatings/VehicleId/{vid}")
            res = (r.json().get("Results") or [{}])[0]
            return {"stars": int(res["OverallRating"]),  # "Not Rated" -> except
                    "vehicle": res.get("VehicleDescription", ""),
                    "url": "https://www.nhtsa.gov/ratings"}
    except Exception:
        return None


def _valid_stats(stats: dict) -> bool:
    """A usable stock baseline: all nine ratings present, in range, not all
    zero, and real hp/0-60/top-speed figures. Anything else is a partial or
    hallucinated reply and must never be cached (issue #28)."""
    try:
        ratings = [float(stats[k]) for k in BAR_STATS]
        figures = [float(stats[k])
                   for k in ("hp", "zero_to_sixty", "top_speed_mph")]
    except (KeyError, TypeError, ValueError):
        return False
    return (all(0 <= r <= 100 for r in ratings) and any(ratings)
            and all(f > 0 for f in figures))


async def _generate_stats(car: dict) -> dict | None:
    """The car's STOCK baseline (mods compose on top via catalog deltas),
    with the safety score grounded in the car's NHTSA 5-Star rating when one
    exists. An invalid block is rejected so the next enrichment retries."""
    desc = _car_desc(car)
    if not desc:
        return None
    nhtsa = await _nhtsa_rating(car.get("year"))
    grounding = (
        f"NHTSA rates the {nhtsa['vehicle']} {nhtsa['stars']}/5 stars overall "
        "— anchor the safety score to that."
        if nhtsa else
        "this model year has no NHTSA 5-Star rating; score it by its "
        "era-appropriate safety equipment."
    )
    resp = await _stats_llm.ainvoke(
        STATS_PROMPT.format(desc=desc, grounding=grounding))
    m = re.search(r"\{.*\}", resp.content, re.S)
    try:
        stats = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        stats = None
    if not stats or not _valid_stats(stats):
        print(f"stats rejected for {desc}: {resp.content[:200]!r}")
        return None
    if nhtsa:
        stats["nhtsa"] = nhtsa  # shown in the UI with a link to nhtsa.gov
    stats["fp"] = _stats_fp(car)  # cache key: recompute only when this drifts
    return stats


def _image_prompt(car: dict) -> str:
    color = car.get("color") or "a factory paint color appropriate to that generation"
    return (
        f"Photorealistic studio photograph, full side profile, of a {_car_desc(car)} "
        f"Ford Mustang in {color}. Accurate body style, proportions, and trim "
        "details for that exact model year and generation. Dark seamless "
        "background, simple soft floor shadow, even studio lighting. "
        "No people, no text, no watermarks."
    )


async def _generate_image(user_id: str, car: dict) -> None:
    """The car's one seed portrait (gpt-image-1), for a car with no portrait
    row at all. DO NOTHING on conflict: if a row appeared mid-generation (a
    user upload, or a concurrent seed) that row wins — a generated image must
    never overwrite anything (issues #31, #34)."""
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
            "INSERT INTO car_images (user_id, car_id, image, prompt) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (user_id, car_id) DO NOTHING",
            (user_id, car["id"], image, prompt),
        )


async def _sync_portrait(user_id: str, car: dict) -> None:
    """Seed the car's portrait when none exists; any existing row (generated
    or user-uploaded) freezes it forever — no identity, color, or mod change
    ever spends another image-model call (issue #34)."""
    async with await _db() as conn:
        row = await (
            await conn.execute(
                "SELECT 1 FROM car_images WHERE user_id = %s AND car_id = %s",
                (user_id, car["id"]),
            )
        ).fetchone()
    if _portrait_action(row) == "generate":
        await _generate_image(user_id, car)


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
    remove_mods: list[str] | None = None,
    remove_wishlist: list[str] | None = None,
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
    arguments optional; new values merge into the existing profile.
    When the user says a mod or wishlist item is GONE ("I took the intake
    off", "drop the exhaust from my wishlist"), pass remove_mods /
    remove_wishlist with the user's words for the item. "I installed the X
    from my wishlist" = remove_wishlist=["X"] plus mods=["X"] in the SAME
    call; "replaced X with Y" = remove_mods=["X"] plus mods=["Y"]. NEVER
    confirm a removal without calling this tool."""
    user_id = config["configurable"]["user_id"]
    updates = {
        k: v
        for k, v in dict(year=year, trim=trim, generation=generation,
                         color=color, nickname=nickname,
                         mods=mods, wishlist=wishlist).items()
        if v is not None
    }
    removals = {k: v for k, v in dict(mods=remove_mods,
                                      wishlist=remove_wishlist).items() if v}
    missed: list[str] = []
    full = False
    async with await _db() as conn:
        # ponytail: read-merge-write, no row lock; fine for one-user-per-thread chat
        profile = await _load_profile(conn, user_id)
        cars = profile.setdefault("cars", [])
        if updates or car or removals:
            target = _match_car(cars, car, updates)
            if target is None and not updates:  # removal-only, car unknown
                return "No matching car in the garage; nothing removed."
            if target is None and len(cars) >= MAX_CARS:
                full = True  # skip the car; goals (if any) still merge below
            else:
                if target is None:
                    target = {"id": uuid.uuid4().hex[:8]}
                    cars.append(target)
                missed = _apply_car_updates(target, updates, removals)
                _autofill_generation(target)
                if target.get("stats") and target["stats"].get("fp") != _stats_fp(target):
                    target["stats"] = None  # identity changed -> recompute stock baseline
        if goals:
            current = profile.get("goals", [])
            profile["goals"] = current + [g for g in goals if g not in current]
        # keep the row's onboarded flag, not this read's possibly-stale copy:
        # the agent may run complete_onboarding in parallel with this call,
        # and writing stale false back would re-lock a just-unlocked user
        await conn.execute(
            "INSERT INTO garage (user_id, profile) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET profile = "
            "CASE WHEN garage.profile ? 'onboarded' THEN EXCLUDED.profile "
            "|| jsonb_build_object('onboarded', garage.profile->'onboarded') "
            "ELSE EXCLUDED.profile END",
            (user_id, Json(profile)),
        )
    if full:
        return (f"{GARAGE_FULL_MSG}. The new car was NOT saved — tell the "
                "user they must delete a car from their garage before "
                "adding another.")
    if missed:
        return ("Garage profile updated, but these weren't in the garage "
                f"(nothing removed): {', '.join(missed)}. Tell the user "
                "instead of confirming the removal.")
    return "Garage profile updated."


@tool
async def complete_onboarding(config: RunnableConfig) -> str:
    """Mark first-run onboarding finished. Call exactly once, right after the
    user answers the final onboarding question (or declines onboarding) —
    never at any other time."""
    user_id = config["configurable"]["user_id"]
    async with await _db() as conn:
        # surgical jsonb merge, no read: the agent often calls this in the
        # same turn as the final update_garage, and a read-merge-write here
        # would clobber that parallel write (lost the last goal in testing)
        await conn.execute(
            "INSERT INTO garage (user_id, profile) "
            "VALUES (%s, '{\"onboarded\": true}') "
            "ON CONFLICT (user_id) DO UPDATE "
            "SET profile = garage.profile || '{\"onboarded\": true}'",
            (user_id,),
        )
    return "Onboarding recorded. Thank the user and let them know their profile is saved."


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
                # existing installs may still carry identity_fp/build_fp
                # columns from the retired portrait-edit path (issue #34);
                # they're unused and harmless
                "CREATE TABLE IF NOT EXISTS car_images ("
                "user_id TEXT NOT NULL, car_id TEXT NOT NULL, "
                "image BYTEA NOT NULL, prompt TEXT NOT NULL, "
                "user_uploaded BOOLEAN NOT NULL DEFAULT FALSE, "
                "content_type TEXT, "
                "PRIMARY KEY (user_id, car_id))"
            )
            await conn.execute(  # user-photo columns (issue #31)
                "ALTER TABLE car_images ADD COLUMN IF NOT EXISTS "
                "user_uploaded BOOLEAN NOT NULL DEFAULT FALSE"
            )
            await conn.execute(
                "ALTER TABLE car_images ADD COLUMN IF NOT EXISTS content_type TEXT"
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
             update_garage, update_instructions, complete_onboarding],
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


async def _heartbeat(gen, interval: float | None = None):
    """Pass gen through, inserting a ping event whenever `interval` seconds
    pass without an item, so clients can tell a slow tool call from a dead
    connection (issue #26)."""
    if interval is None:
        interval = float(os.environ.get("CHAT_PING_SECONDS", "10"))
    it = aiter(gen)
    try:
        nxt = asyncio.ensure_future(anext(it))
        while True:
            done, _ = await asyncio.wait({nxt}, timeout=interval)
            if not done:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                continue
            try:
                item = nxt.result()
            except StopAsyncIteration:
                return
            yield item
            nxt = asyncio.ensure_future(anext(it))
    finally:
        nxt.cancel()


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
        async with await _db() as conn:
            rows = await (
                await conn.execute(
                    "SELECT car_id FROM car_images "
                    "WHERE user_id = %s AND user_uploaded",
                    (user_id,),
                )
            ).fetchall()
        uploaded = {r[0] for r in rows}
        # composed current/dream bars ride along so clients never re-derive
        # them; photo_uploaded drives the upload-pill visibility (issue #37)
        profile = {**profile,
                   "cars": [{**c, "bars": _compose_bars(c),
                             "photo_uploaded": c["id"] in uploaded}
                            for c in profile["cars"]]}
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
                "SELECT image, content_type FROM car_images "
                "WHERE user_id = %s AND car_id = %s",
                (user_id, car_id),
            )
        ).fetchone()
    if not row:
        raise HTTPException(404, "portrait not generated yet")
    return Response(
        content=bytes(row[0]),
        media_type=row[1] or "image/png",  # uploads keep their own type (#31)
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.put("/garage/{user_id}/cars/{car_id}/image", status_code=204)
async def upload_car_image(user_id: str, car_id: str, request: Request,
                           auth_uid: AuthUid = None):
    """Replace the portrait with the user's own photo — raw image bytes in the
    body (no multipart). An uploaded photo is canonical: background enrichment
    never overwrites it; re-uploading replaces it (issue #31)."""
    user_id = auth_uid or user_id
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip()
    if not ctype.startswith("image/"):
        raise HTTPException(415, "send an image file")
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty image")
    if len(body) > 8_000_000:
        raise HTTPException(413, "image must be under 8 MB")
    async with await _db() as conn:
        profile = await _load_profile(conn, user_id)
        if not any(c["id"] == car_id for c in profile.get("cars", [])):
            raise HTTPException(404, "car not found")
        await conn.execute(
            "INSERT INTO car_images (user_id, car_id, image, prompt, "
            "content_type, user_uploaded) "
            "VALUES (%s, %s, %s, 'user upload', %s, TRUE) "
            "ON CONFLICT (user_id, car_id) DO UPDATE SET "
            "image = EXCLUDED.image, prompt = EXCLUDED.prompt, "
            "content_type = EXCLUDED.content_type, user_uploaded = TRUE",
            (user_id, car_id, body, ctype),
        )


@app.get("/garage/{user_id}/cars/{car_id}/shop")
async def upgrade_shop(user_id: str, car_id: str, auth_uid: AuthUid = None):
    """The car's Upgrade Shop: a recommended strip of 2-3 eligible sponsor
    products (category fit against this car's generation, current mods, and
    wishlist gaps) plus the full catalog — sponsor products and generic mod
    categories — each row carrying the nine-stat delta chips for this car's
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

    # The browsable catalog: specific installable sponsor products + generic
    # mod categories. Non-recommendable ad campaigns, services, and broad
    # product lines are not garage upgrades (the latter stay chat-only).
    rows = [row(e) for e in CATALOG
            if (e["recommendable"] and e.get("specific")) or not e["sponsored"]]
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
        if len(cars) >= MAX_CARS:
            raise HTTPException(400, GARAGE_FULL_MSG)
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
            elif isinstance(v, list):
                target[k] = _dedupe(v)  # retried adds must not double-store (#29)
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
                          if m.type == "human" and _msg_text(m)
                          and _msg_text(m) != ONBOARD_KICKOFF),
                         "Earlier conversation")
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
        if not text or text == ONBOARD_KICKOFF:  # the hidden onboarding trigger
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
    # First-run onboarding (issue #46): the kickoff sentinel from a user we
    # know nothing about stamps onboarded=false; the flag staying false keeps
    # the interview script in the prompt across turns (even once the garage
    # fills up mid-interview) until complete_onboarding flips it true.
    if (req.message == ONBOARD_KICKOFF and "onboarded" not in profile
            and not profile.get("cars") and not profile.get("goals")):
        profile["onboarded"] = False
        async with await _db() as conn:
            await conn.execute(
                "INSERT INTO garage (user_id, profile) VALUES (%s, %s) "
                "ON CONFLICT (user_id) DO UPDATE SET profile = EXCLUDED.profile",
                (user_id, Json(profile)),
            )
    async with await _db() as conn:  # title = first user message, then just bump
        await conn.execute(
            "INSERT INTO chats (user_id, chat_id, title) VALUES (%s, %s, %s) "
            "ON CONFLICT (user_id, chat_id) DO UPDATE SET updated_at = now()",
            (user_id, req.chat_id,
             "Welcome" if req.message == ONBOARD_KICKOFF else req.message[:60]),
        )
    system = SYSTEM_PROMPT
    if profile.get("onboarded") is False:
        system += ONBOARDING_PROMPT
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
        try:
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
                elif isinstance(msg, AIMessageChunk):
                    # The model has decided to call a tool: tell the client
                    # which one, so it can show status instead of bare dots.
                    for tc in (msg.tool_call_chunks or []):
                        if tc.get("name"):
                            yield f"data: {json.dumps({'type': 'tool_start', 'name': tc['name']})}\n\n"
                    if isinstance(msg.content, str) and msg.content:
                        answer_parts.append(msg.content)
                        yield f"data: {json.dumps({'type': 'token', 'text': msg.content})}\n\n"
        except Exception:
            # Real error goes to the server log only; the client gets a
            # generic event it can render as a friendly message (issue #26).
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error'})}\n\n"
            yield "data: [DONE]\n\n"
            return
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _heartbeat(sse()),
        media_type="text/event-stream",
        background=BackgroundTask(
            _post_turn, user_id, req.session_id, req.message, answer_parts
        ),
    )
