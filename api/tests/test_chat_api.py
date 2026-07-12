"""Seam-1: ASGI transport against the real app — real Postgres checkpointer
(docker compose), real in-memory index, real LLM via the gateway."""

import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import app


async def collect_events(client, message, user_id, session_id=None):
    events = []
    body = {"message": message, "user_id": user_id}
    if session_id:
        body["session_id"] = session_id
    async with client.stream("POST", "/chat", json=body, timeout=120) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[len("data: "):]))
    return events


@pytest.mark.asyncio
async def test_health_and_chat_turn():
    async with app.router.lifespan_context(app):  # ASGITransport skips lifespan
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.json() == {"status": "ok"}

            events = await collect_events(
                client, "Tell me about the 1971-1973 Mustangs", str(uuid.uuid4())
            )
            answer = "".join(e["text"] for e in events if e["type"] == "token")
            (citations_event,) = [e for e in events if e["type"] == "citations"]

            assert len(answer) > 50
            assert citations_event["citations"], "expected archive citations"
            for c in citations_event["citations"]:
                assert c["title"]
                assert c["url"].startswith("https://www.mustangdriver.com/")


@pytest.mark.asyncio
async def test_stream_carries_tool_start_and_heartbeat(monkeypatch):
    """Issue #26: a tool-using question announces the tool the moment the
    model calls it (tool_start, before the tool result), and silent gaps
    carry ping events so clients can tell a slow tool from a dead socket."""
    monkeypatch.setenv("CHAT_PING_SECONDS", "0.2")
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            events = await collect_events(
                client, "Search the archive: what did MustangDriver write "
                "about the Boss 302?", str(uuid.uuid4())
            )
            types = [e["type"] for e in events]
            assert "tool_start" in types, types
            assert types.index("tool_start") < types.index("tool"), types
            assert "ping" in types, types  # the LLM thinks far longer than 0.2s
            started = [e["name"] for e in events if e["type"] == "tool_start"]
            assert "search_archive" in started, started


@pytest.mark.asyncio
async def test_stream_error_is_generic_event_not_traceback(monkeypatch):
    """A mid-stream failure must reach the client as a bare error event —
    the real exception stays in the server log."""
    import app as app_module

    class ExplodingAgent:
        async def astream(self, *a, **kw):
            raise RuntimeError("secret internal detail")
            yield  # pragma: no cover — makes this an async generator

    async with app.router.lifespan_context(app):
        # patch after lifespan startup: it rebuilds the real _agent
        monkeypatch.setattr(app_module, "_agent", ExplodingAgent())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            events = await collect_events(client, "hi", str(uuid.uuid4()))
            assert {"type": "error"} in events, events
            assert "secret internal detail" not in json.dumps(events)
