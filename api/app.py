"""Ask MustangDriver chat API.

FastAPI app: POST /chat streams SSE tokens + a citations event from a
LangGraph ReAct agent with one tool (search_archive over an in-memory Qdrant
collection built from index_artifact/ at startup). Conversation history keyed
by user_id via the Postgres checkpointer (docker compose, repo root).

Run locally:  uv run uvicorn app:app --port 8000
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.tools import tool
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

Answer questions using the search_archive tool and ground every answer in the \
retrieved articles. Cite your sources inline as markdown links using each \
article's title and URL, e.g. [Article Title](https://www.mustangdriver.com/...). \
If the archive doesn't cover a topic, say so plainly rather than guessing."""

_embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url=GATEWAY_URL,
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    check_embedding_ctx_length=False,  # gateway wants raw strings, not token arrays
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qdrant, _agent
    _qdrant = build_index()
    async with AsyncPostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as saver:
        await saver.setup()  # idempotent
        _agent = create_react_agent(
            ChatOpenAI(
                model="anthropic/claude-sonnet-4.5",
                base_url=GATEWAY_URL,
                api_key=os.environ["AI_GATEWAY_API_KEY"],
            ),
            [search_archive],
            prompt=SYSTEM_PROMPT,
            checkpointer=saver,
        )
        yield


app = FastAPI(lifespan=lifespan)
# ponytail: open CORS, no auth on this API anyway; restrict to the web origin if that changes
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    user_id: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    async def sse():
        citations, seen = [], set()
        stream = _agent.astream(
            {"messages": [{"role": "user", "content": req.message}]},
            {"configurable": {"thread_id": req.user_id}},
            stream_mode="messages",
        )
        async for msg, _meta in stream:
            if isinstance(msg, ToolMessage) and msg.name == "search_archive":
                for hit in json.loads(msg.content):
                    if hit["url"] not in seen:
                        seen.add(hit["url"])
                        citations.append({"title": hit["title"], "url": hit["url"]})
            elif isinstance(msg, AIMessageChunk) and isinstance(msg.content, str) and msg.content:
                yield f"data: {json.dumps({'type': 'token', 'text': msg.content})}\n\n"
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
