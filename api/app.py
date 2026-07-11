"""Ask MustangDriver chat API.

FastAPI app: POST /chat streams SSE tokens + tool/citations events from a
LangGraph ReAct agent with three tools: search_archive (in-memory Qdrant over
index_artifact/, built at startup), web_search (Tavily, live web), and
check_recalls (NHTSA Recalls API), plus two memory tools: update_garage
(semantic: the user's car, mods, wishlist, goals) and update_instructions
(procedural: standing answer preferences), both keyed by user_id in Postgres
and folded into the system prompt each turn. Episodic memory: after each turn
a background task has Claude Haiku keep a rolling 2-3 sentence summary of the
session (keyed by user_id + session_id from the client), and recent past-session
summaries are injected into the system prompt. GET /garage/{user_id} returns
everything known. Conversation history keyed by user_id via the Postgres
checkpointer (docker compose, repo root).

Run locally:  uv run uvicorn app:app --port 8000
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from psycopg.types.json import Json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
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
- Whenever the user mentions facts about their own car — model year, trim, \
generation (e.g. S550), installed mods, wishlist/planned mods, or goals for \
the car (track, show, daily driver) — silently call update_garage with those \
facts. Never announce or mention that you recorded anything.
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


async def _get_memory(user_id: str) -> tuple[dict, list[str], list[dict]]:
    """(garage profile, standing instructions, recent session summaries) for a
    user; empty when unknown. Summaries are recent-first, with session_id so
    /chat can exclude the session in progress (limit 6 = 5 past + current)."""
    async with await _db() as conn:
        row = await (
            await conn.execute("SELECT profile FROM garage WHERE user_id = %s", (user_id,))
        ).fetchone()
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
    return (row[0] if row else {}), [r[0] for r in rows], summaries


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


@tool
async def update_garage(
    config: RunnableConfig,
    year: int | None = None,
    trim: str | None = None,
    generation: str | None = None,
    mods: list[str] | None = None,
    wishlist: list[str] | None = None,
    goals: list[str] | None = None,
) -> str:
    """Record facts the user reveals about their own Mustang: model year, trim
    (e.g. GT Premium), generation (e.g. S550), installed mods, wishlist/planned
    mods, and goals (track, show, daily driver). All arguments optional; new
    values merge into the existing profile."""
    user_id = config["configurable"]["user_id"]
    updates = {
        k: v
        for k, v in dict(year=year, trim=trim, generation=generation,
                         mods=mods, wishlist=wishlist, goals=goals).items()
        if v is not None
    }
    async with await _db() as conn:
        # ponytail: read-merge-write, no row lock; fine for one-user-per-thread chat
        row = await (
            await conn.execute("SELECT profile FROM garage WHERE user_id = %s", (user_id,))
        ).fetchone()
        profile = row[0] if row else {}
        for k, v in updates.items():
            if isinstance(v, list):
                current = profile.get(k, [])
                profile[k] = current + [x for x in v if x not in current]
            else:
                profile[k] = v
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
    return {
        "profile": profile,
        "instructions": instructions,
        "summaries": [{"summary": s["summary"], "date": s["date"]} for s in summaries],
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    profile, instructions, summaries = await _get_memory(req.user_id)
    system = SYSTEM_PROMPT
    if profile:
        system += f"\n\nThe user's garage profile (their car):\n{json.dumps(profile)}"
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
            _summarize, req.user_id, req.session_id, req.message, answer_parts
        ),
    )
