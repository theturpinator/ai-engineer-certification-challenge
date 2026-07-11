"""Seam-1: memory flows — real LLM, real Postgres, ASGI transport.

Car facts mentioned in chat land in the garage profile; a standing
preference stated once is stored and reflected in later answers.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import app
from test_chat_api import collect_events


@pytest.mark.asyncio
async def test_car_extraction_lands_in_garage():
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Unknown user: empty structures, not 404.
            resp = await client.get(f"/garage/{user_id}")
            assert resp.status_code == 200
            assert resp.json() == {"profile": {}, "instructions": [], "summaries": []}

            await collect_events(
                client,
                "I drive a 2016 Mustang GT Premium, S550 generation, with a "
                "cold air intake installed. I want to add a supercharger "
                "eventually. It's my weekend track car.",
                user_id,
            )

            profile = (await client.get(f"/garage/{user_id}")).json()["profile"]
            assert int(profile["year"]) == 2016, profile
            assert "gt" in str(profile["trim"]).lower(), profile
            assert "s550" in str(profile["generation"]).lower(), profile
            assert profile.get("mods"), profile
            assert profile.get("wishlist"), profile
            assert profile.get("goals"), profile


@pytest.mark.asyncio
async def test_preference_round_trip():
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await collect_events(
                client,
                "From now on, always end every answer with the word CHEERS.",
                user_id,
            )

            instructions = (await client.get(f"/garage/{user_id}")).json()["instructions"]
            assert any("cheers" in i.lower() for i in instructions), instructions

            events = await collect_events(
                client, "What generation is the 1965 Mustang?", user_id
            )
            answer = "".join(e["text"] for e in events if e["type"] == "token")
            assert "cheers" in answer.lower()[-100:], answer
