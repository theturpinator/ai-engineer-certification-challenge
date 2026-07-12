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
            cars = profile.get("cars") or []
            assert len(cars) == 1, profile
            car = cars[0]
            assert car["id"], car
            assert int(car["year"]) == 2016, car
            assert "gt" in str(car["trim"]).lower(), car
            assert "s550" in str(car["generation"]).lower(), car
            assert car.get("mods"), car
            assert car.get("wishlist"), car
            assert profile.get("goals"), profile  # goals stay user-level


@pytest.mark.asyncio
async def test_partial_car_mention_lands_same_turn():
    """The production repro: a partial mention (trim + color, no year) must
    create a garage car in the SAME turn; the year arriving later updates
    that same car and derives the generation."""
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await collect_events(
                client,
                "Hey what's good! My name is Brendan and I have a blue mustang GT",
                user_id,
            )
            cars = (await client.get(f"/garage/{user_id}")).json()["profile"].get("cars") or []
            assert len(cars) == 1, cars
            car = cars[0]
            assert "gt" in str(car.get("trim", "")).lower(), car
            assert "blue" in str(car.get("color", "")).lower(), car

            await collect_events(client, "Oh, it's a 2016 by the way.", user_id)
            cars2 = (await client.get(f"/garage/{user_id}")).json()["profile"]["cars"]
            assert len(cars2) == 1, cars2  # same car updated, not a second entry
            assert cars2[0]["id"] == car["id"], cars2
            assert int(cars2[0]["year"]) == 2016, cars2
            assert cars2[0].get("generation") == "S550", cars2  # derived


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
