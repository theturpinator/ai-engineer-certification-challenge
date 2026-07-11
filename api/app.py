"""Ask MustangDriver chat API.

FastAPI app: POST /chat streams SSE tokens + tool/citations events from a
LangGraph ReAct agent with three tools: search_archive (in-memory Qdrant over
index_artifact/, built at startup), web_search (Tavily, live web), and
check_recalls (NHTSA Recalls API), plus two memory tools: update_garage
(semantic: the user's cars, per-car mods/wishlist, user-level goals) and
update_instructions (procedural: standing answer preferences), both keyed by
user_id in Postgres and folded into the system prompt each turn. Episodic
memory: after each turn a background task has Claude Haiku keep a rolling 2-3
sentence summary of the session (keyed by user_id + session_id from the
client), and recent past-session summaries are injected into the system
prompt. The garage holds multiple cars (profile.cars, legacy flat profiles
migrate on read); each car is enriched in the background with arcade-style
stock stats (Sonnet, cached in the car) and an AI-generated portrait
(gpt-image-1 via the gateway, cached in car_images). GET /garage/{user_id}
returns everything known; cars are editable via PATCH/DELETE
/garage/{user_id}/cars/{car_id}. Conversation history keyed by user_id via
the Postgres checkpointer (docker compose, repo root).

Run locally:  uv run uvicorn app:app --port 8000
"""

import asyncio
import base64
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
import numpy as np
import psycopg
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
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

If no tool can answer, say so plainly rather than guessing.

Memory policy:
- Whenever the user mentions facts about one of their own cars — model year, \
trim, generation (e.g. S550), color, nickname, installed mods, \
wishlist/planned mods, or goals (track, show, daily driver) — silently call \
update_garage with those facts. The garage holds multiple cars: pass the \
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
    if any(updates.get(k) for k in ("year", "trim", "generation")):
        return None  # identifying info matching no car -> new entry
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


STATS_PROMPT = """You are a Ford Mustang encyclopedia. For the completely STOCK \
{desc} Ford Mustang (factory condition, no modifications), reply with ONLY this \
JSON object, no other text:
{{"power": <0-100>, "acceleration": <0-100>, "top_speed": <0-100>, \
"handling": <0-100>, "braking": <0-100>, "hp": <stock horsepower, integer>, \
"zero_to_sixty": <stock 0-60 mph time in seconds, float>, \
"top_speed_mph": <stock top speed in mph, integer>}}
The 0-100 values are arcade-racing-game ratings calibrated across the entire \
Mustang range, 1964 to today: a 1974 Mustang II is roughly 15-25 power, a base \
1990s V6 around 30-40, a 2015+ GT around 75-85, a 2020 Shelby GT500 is 95-100."""


async def _generate_stats(car: dict) -> dict | None:
    desc = _car_desc(car)
    if not desc:
        return None
    resp = await _stats_llm.ainvoke(STATS_PROMPT.format(desc=desc))
    m = re.search(r"\{.*\}", resp.content, re.S)
    return json.loads(m.group()) if m else None


def _image_prompt(car: dict) -> str:
    color = car.get("color") or "a factory paint color appropriate to that generation"
    return (
        f"Photorealistic studio photograph, full side profile, of a {_car_desc(car)} "
        f"Ford Mustang in {color}. Body style exactly accurate for that model year "
        "and generation. Dark seamless studio background, soft reflective floor, "
        "dramatic rim lighting. No people, no text, no watermarks."
    )


async def _generate_image(user_id: str, car: dict) -> None:
    """Generate + cache the car portrait unless the cached one was made from
    the same prompt (the prompt doubles as the identity/color fingerprint)."""
    prompt = _image_prompt(car)
    async with await _db() as conn:
        row = await (
            await conn.execute(
                "SELECT prompt FROM car_images WHERE user_id = %s AND car_id = %s",
                (user_id, car["id"]),
            )
        ).fetchone()
    if row and row[0] == prompt:
        return
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
            "VALUES (%s, %s, %s, %s) ON CONFLICT (user_id, car_id) DO UPDATE "
            "SET image = EXCLUDED.image, prompt = EXCLUDED.prompt",
            (user_id, car["id"], image, prompt),
        )


async def _save_car_stats(user_id: str, car_id: str, stats: dict) -> None:
    """Re-read-modify-write so a concurrent chat turn's garage write between
    enrichment start and finish isn't clobbered."""
    async with await _db() as conn:
        profile = await _load_profile(conn, user_id)
        for c in profile.get("cars", []):
            if c["id"] == car_id and not c.get("stats"):
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
            if not car.get("stats"):
                stats = await _generate_stats(car)
                if stats:
                    await _save_car_stats(user_id, car["id"], stats)
            await _generate_image(user_id, car)
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
    """Record facts the user reveals about their own Mustang(s). The garage
    holds multiple cars. Pass `car` with the user's own words for which car
    they mean (e.g. "my 2016 GT", "the Fox-body") whenever they own more than
    one or mention a car not yet in their garage — an unknown car automatically
    becomes a new garage entry. Per-car facts: model year, trim (e.g. GT
    Premium), generation (e.g. S550), color, nickname, installed mods, and
    wishlist/planned mods. goals (track, show, daily driver) apply to the user
    as a whole. All arguments optional; new values merge into the existing
    profile."""
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
            if any(str(target.get(k)) != str(updates[k])
                   for k in ("year", "trim", "generation") if k in updates):
                target["stats"] = None  # identity changed -> stats stale
            for k, v in updates.items():
                if isinstance(v, list):
                    current = target.get(k, [])
                    target[k] = current + [x for x in v if x not in current]
                else:
                    target[k] = v
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
                "PRIMARY KEY (user_id, car_id))"
            )
        _agent = create_react_agent(
            ChatOpenAI(
                model="anthropic/claude-sonnet-4.5",
                base_url=GATEWAY_URL,
                api_key=os.environ["AI_GATEWAY_API_KEY"],
            ),
            [search_archive, web_search, check_recalls, update_garage, update_instructions],
            prompt=_prompt,
            checkpointer=saver,
        )
        yield


app = FastAPI(lifespan=lifespan)
# ponytail: open CORS, no auth on this API anyway; restrict to the web origin if that changes
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: str = "default"  # old clients without one all share this bucket


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/garage/{user_id}")
async def garage(user_id: str):
    profile, instructions, summaries = await _get_memory(user_id)
    if profile.get("cars"):
        # opportunistic fire-and-forget: fills stats/portraits missed earlier;
        # _enrich_garage no-ops cheaply when everything is current
        asyncio.get_running_loop().create_task(_enrich_garage(user_id))
    return {
        "profile": profile,
        "instructions": instructions,
        "summaries": [{"summary": s["summary"], "date": s["date"]} for s in summaries],
    }


@app.get("/garage/{user_id}/cars/{car_id}/image")
async def car_image(user_id: str, car_id: str):
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


@app.patch("/garage/{user_id}/cars/{car_id}")
async def patch_car(user_id: str, car_id: str, patch: CarPatch,
                    background_tasks: BackgroundTasks):
    updates = patch.model_dump(exclude_unset=True)
    async with await _db() as conn:
        profile = await _load_profile(conn, user_id)
        target = next((c for c in profile.get("cars", []) if c["id"] == car_id), None)
        if target is None:
            raise HTTPException(404, "car not found")
        if any(str(target.get(k)) != str(updates[k])
               for k in ("year", "trim", "generation") if k in updates):
            target["stats"] = None  # identity changed -> regenerate
        for k, v in updates.items():
            if v is None:
                target.pop(k, None)
            else:
                target[k] = v
        await conn.execute(
            "UPDATE garage SET profile = %s WHERE user_id = %s",
            (Json(profile), user_id),
        )
    background_tasks.add_task(_enrich_garage, user_id)
    return target


@app.delete("/garage/{user_id}/cars/{car_id}", status_code=204)
async def delete_car(user_id: str, car_id: str):
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


async def _post_turn(user_id: str, session_id: str, user_text: str, parts: list[str]):
    """After the SSE stream: update the session summary, then fill in any
    missing car stats/portraits the turn's garage writes created."""
    await _summarize(user_id, session_id, user_text, parts)
    await _enrich_garage(user_id)


@app.post("/chat")
async def chat(req: ChatRequest):
    profile, instructions, summaries = await _get_memory(req.user_id)
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
        stream = _agent.astream(
            {"messages": [{"role": "user", "content": req.message}]},
            {"configurable": {
                "thread_id": req.user_id,
                "user_id": req.user_id,
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
            elif isinstance(msg, AIMessageChunk) and isinstance(msg.content, str) and msg.content:
                answer_parts.append(msg.content)
                yield f"data: {json.dumps({'type': 'token', 'text': msg.content})}\n\n"
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        background=BackgroundTask(
            _post_turn, req.user_id, req.session_id, req.message, answer_parts
        ),
    )
