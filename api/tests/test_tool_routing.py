"""Tool-routing smoke suite: real HTTP seam, real LLM, real tools.

Asserts the agent routes recall questions to check_recalls, live-web
questions to web_search (with a disclosure in the answer), and archive
questions to search_archive (with citations) — observed via the `tool`
SSE events.
"""

import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import app


async def ask(client, message):
    events = []
    async with client.stream(
        "POST", "/chat", json={"message": message, "user_id": str(uuid.uuid4())},
        timeout=180,
    ) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[len("data: "):]))
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    tools = [e["name"] for e in events if e["type"] == "tool"]
    (citations_event,) = [e for e in events if e["type"] == "citations"]
    return answer, tools, citations_event["citations"]


RECALL_QUESTIONS = [
    "Are there any recalls on my 2020 Mustang?",
    "Has NHTSA issued any safety recalls for the 2015 Mustang?",
]
LIVE_QUESTIONS = [
    "What does a used 2020 Mustang GT go for right now?",
    "What is the latest Ford Mustang news this week?",
]
ARCHIVE_QUESTIONS = [
    "Tell me about the 1971-1973 Mustangs",
    "Where were the first Mustangs built?",
]


@pytest.mark.asyncio
async def test_tool_routing():
    async with app.router.lifespan_context(app):  # ASGITransport skips lifespan
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for q in RECALL_QUESTIONS:
                answer, tools, _ = await ask(client, q)
                assert "check_recalls" in tools, f"{q!r} routed to {tools}"
                assert len(answer) > 50

            for q in LIVE_QUESTIONS:
                answer, tools, _ = await ask(client, q)
                assert "web_search" in tools, f"{q!r} routed to {tools}"
                lower = answer.lower()
                assert "web search" in lower or "live web" in lower, (
                    f"{q!r} answer lacks live-web disclosure: {answer[:200]}"
                )

            for q in ARCHIVE_QUESTIONS:
                answer, tools, citations = await ask(client, q)
                assert "search_archive" in tools, f"{q!r} routed to {tools}"
                assert "web_search" not in tools, f"{q!r} hit the web: {tools}"
                assert citations, f"{q!r} returned no citations"
                assert len(answer) > 50
