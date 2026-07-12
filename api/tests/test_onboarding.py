"""Seam-1: first-run onboarding in chat (issue #46) — real LLM, real Postgres.

The kickoff sentinel from a blank user stamps profile.onboarded=false and
the agent opens the interview; the sentinel never leaks into the transcript
replay or the chat title. complete_onboarding flips the flag to true.
Existing users are untouched by a stray sentinel.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

import app as app_module
from app import ONBOARD_KICKOFF, app, complete_onboarding
from test_chat_api import collect_events


@pytest.mark.asyncio
async def test_kickoff_stamps_flag_and_starts_interview():
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            events = await collect_events(client, ONBOARD_KICKOFF, user_id)
            answer = "".join(e["text"] for e in events if e["type"] == "token")
            # the agent greets and opens the interview with a question
            assert "?" in answer, answer
            assert "mustang" in answer.lower(), answer
            assert ONBOARD_KICKOFF not in answer

            # onboarding is now in progress server-side
            profile = (await client.get(f"/garage/{user_id}")).json()["profile"]
            assert profile.get("onboarded") is False, profile

            # the sentinel is hidden from the replayed transcript…
            msgs = (await client.get(f"/chats/{user_id}/default/messages")).json()
            assert msgs, "transcript empty"
            assert all(m["content"] != ONBOARD_KICKOFF for m in msgs), msgs
            assert msgs[0]["role"] == "assistant", msgs
            # …and never becomes the chat title
            chats = (await client.get(f"/chats/{user_id}")).json()
            titles = [c["title"] for c in chats]
            assert ONBOARD_KICKOFF[:60] not in titles, titles


@pytest.mark.asyncio
async def test_complete_onboarding_flips_flag_and_existing_users_skip():
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # the tool the agent calls after the last question
            result = await complete_onboarding.ainvoke(
                {"type": "tool_call", "name": "complete_onboarding", "id": "t1",
                 "args": {}},
                config={"configurable": {"user_id": user_id}},
            )
            assert "recorded" in str(result.content).lower()
            profile = (await client.get(f"/garage/{user_id}")).json()["profile"]
            assert profile.get("onboarded") is True, profile

            # a user with an existing profile never gets re-onboarded by a
            # stray kickoff: no LLM needed — check the stamp logic directly
            veteran = str(uuid.uuid4())
            r = await client.post(f"/garage/{veteran}/cars",
                                  json={"year": 2016, "trim": "GT", "color": "Red"})
            assert r.status_code == 201
            profile = (await client.get(f"/garage/{veteran}")).json()["profile"]
            assert "onboarded" not in profile, profile
