"""Seam-1: multi-chat — real LLM + Postgres checkpointer, ASGI transport.

Turn 1 uses no chat_id (the legacy single-thread client). Turn 2 opens a
second chat. The chat list shows both (legacy thread surfaced as "default"),
the transcript endpoint replays chat 1, and continuing chat 1 still has its
context while chat 2 never saw it.
"""

import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import app


async def send(client, message, user_id, chat_id=None):
    body = {"message": message, "user_id": user_id}
    if chat_id:
        body["chat_id"] = chat_id
    events = []
    async with client.stream("POST", "/chat", json=body, timeout=120) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[len("data: "):]))
    return "".join(e["text"] for e in events if e["type"] == "token")


@pytest.mark.asyncio
async def test_chat_list_switch_and_continue():
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # unknown user: empty list, not 404
            assert (await client.get(f"/chats/{user_id}")).json() == []

            first_msg = "Please remember the word PLATYPUS. Just reply OK."
            await send(client, first_msg, user_id)  # legacy: no chat_id
            await send(client, "What oil should a 2016 GT use?", user_id,
                       chat_id="chat-b")

            chats = (await client.get(f"/chats/{user_id}")).json()
            assert [c["chat_id"] for c in chats] == ["chat-b", "default"], chats
            assert chats[0]["title"].lower().startswith("what oil")
            assert chats[1]["title"] == first_msg[:60]
            assert all(c["updated_at"] for c in chats)

            # reopen chat 1: transcript replayed from the checkpointer
            msgs = (await client.get(f"/chats/{user_id}/default/messages")).json()
            assert msgs[0] == {"role": "user", "content": first_msg}, msgs
            assert msgs[1]["role"] == "assistant" and msgs[1]["content"], msgs

            # continue chat 1 with full context; chat 2 never saw the word
            answer = await send(
                client, "What word did I ask you to remember?", user_id
            )
            assert "platypus" in answer.lower(), answer
            msgs_b = (await client.get(f"/chats/{user_id}/chat-b/messages")).json()
            assert all("platypus" not in m["content"].lower() for m in msgs_b), msgs_b
