"""Seam-1: episodic memory — real LLM + Postgres, ASGI transport.

Session A chats about a distinctive topic; a Haiku-written summary lands in
the summaries table (async, so we poll /garage). Session B (same user, new
session_id) asks what was discussed last time and the answer references it.
"""

import asyncio
import re
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import app
from test_chat_api import collect_events


@pytest.mark.asyncio
async def test_summary_written_and_recalled_next_session():
    user_id = str(uuid.uuid4())
    session_a, session_b = str(uuid.uuid4()), str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await collect_events(
                client,
                "I'm planning to swap a Coyote 5.0 into my 1990 Fox-body. "
                "What should I know before I start?",
                user_id,
                session_id=session_a,
            )

            summaries = []
            for _ in range(30):  # summary write is a background task
                summaries = (await client.get(f"/garage/{user_id}")).json()["summaries"]
                if summaries:
                    break
                await asyncio.sleep(1)
            assert summaries, "no summary written within ~30s"
            summary = summaries[0]["summary"]
            assert summaries[0]["date"]
            assert len(summary) > 20, summary
            sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
            assert 1 <= len(sentences) <= 5, summary

            events = await collect_events(
                client, "What did we talk about last time?", user_id,
                session_id=session_b,
            )
            answer = "".join(e["text"] for e in events if e["type"] == "token")
            assert "coyote" in answer.lower() or "fox" in answer.lower(), answer
