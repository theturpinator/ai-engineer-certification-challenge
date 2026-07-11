"""Seam-1: ASGI transport against the real app — real Postgres checkpointer
(docker compose), real in-memory index, real LLM via the gateway."""

import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import app


async def collect_events(client, message, user_id):
    events = []
    async with client.stream(
        "POST", "/chat", json={"message": message, "user_id": user_id}, timeout=120
    ) as resp:
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
