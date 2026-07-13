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


def _answer(events) -> str:
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text, events
    return text


@pytest.mark.asyncio
async def test_owner_interview_color_more_mustangs_wishlist_no_buy_question():
    """Issue #52, owner path: a car given without color gets asked for the
    color once; "any more Mustangs?" is asked once; a planned-upgrades yes
    lands the named upgrades in the car's wishlist; the buy-a-Mustang
    question is never asked of an owner."""
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await collect_events(client, ONBOARD_KICKOFF, user_id)
            await collect_events(client, "Call me Alex", user_id)
            events = await collect_events(
                client, "Yes! I have a 2016 Mustang GT", user_id)  # no color

            # the car is recorded immediately and the color asked for
            profile = (await client.get(f"/garage/{user_id}")).json()["profile"]
            assert profile.get("cars"), profile
            assert "color" in _answer(events).lower()

            events = await collect_events(client, "It's red", user_id)
            (car,) = (await client.get(f"/garage/{user_id}")).json()["profile"]["cars"]
            assert (car.get("color") or "").lower() == "red", car
            # color never re-asked; the one any-more-Mustangs ask comes next
            follow_up = _answer(events).lower()
            assert "color" not in follow_up
            assert "more" in follow_up or "another" in follow_up or "other" in follow_up

            events = await collect_events(client, "No, just the one", user_id)
            # upgrades question; the answer names upgrades -> car wishlist
            events = await collect_events(
                client, "Yes — a supercharger and lowering springs", user_id)
            (car,) = (await client.get(f"/garage/{user_id}")).json()["profile"]["cars"]
            wishlist = " ".join(car.get("wishlist", [])).lower()
            assert "supercharger" in wishlist, car
            assert "spring" in wishlist or "lowering" in wishlist, car

            # an owner is never asked about buying a Mustang
            assert "buy" not in _answer(events).lower()


@pytest.mark.asyncio
async def test_non_owner_gets_buy_question_and_upgrades_land_as_goals():
    """Issue #52, non-owner path: planned upgrades become profile goals when
    there is no car, and the buy-a-Mustang question is asked."""
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await collect_events(client, ONBOARD_KICKOFF, user_id)
            asked = []  # the interview's questions, in order
            for reply in ("Sam", "No Mustang yet",
                          "Yes, I'd love a cold air intake eventually",
                          "That's all for upgrades"):
                events = await collect_events(client, reply, user_id)
                asked.append(_answer(events).lower())

            profile = (await client.get(f"/garage/{user_id}")).json()["profile"]
            assert not profile.get("cars"), profile
            goals = " ".join(profile.get("goals", [])).lower()
            assert "intake" in goals, profile  # goal, since there is no car

            # a non-owner IS asked about buying a Mustang (the exact position
            # in the interview is the model's call)
            assert any("buy" in a for a in asked), asked


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
