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


@pytest.mark.asyncio
async def test_removal_conversation_removes_mod(monkeypatch):
    """Issue #25: 'I took X off' in chat must end with X gone from the
    garage endpoint's response — no hallucinated confirmations."""
    import app as app_module

    async def fake_enrich(uid):  # stats/portraits covered elsewhere
        return None

    monkeypatch.setattr(app_module, "_enrich_garage", fake_enrich)
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/garage/{user_id}/cars",
                json={"year": 2016, "trim": "GT", "color": "blue"},
            )
            assert resp.status_code == 201, resp.text
            car_id = resp.json()["id"]
            await client.patch(
                f"/garage/{user_id}/cars/{car_id}",
                json={"mods": ["Cold air intake", "Borla exhaust"]},
            )

            await collect_events(
                client, "I took the cold air intake off my GT.", user_id
            )

            profile = (await client.get(f"/garage/{user_id}")).json()["profile"]
            (car,) = profile["cars"]
            mods = [m.lower() for m in car.get("mods", [])]
            assert not any("intake" in m for m in mods), car
            assert any("borla" in m for m in mods), car


@pytest.mark.asyncio
async def test_wishlist_without_identity_never_spawns_a_phantom_car():
    """Issue #52: mods/wishlist facts with no identifying field must not
    create an identity-less car — the tool refuses and points the agent at
    goals instead (a car-less user's planned upgrades stay profile goals)."""
    from app import update_garage

    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            result = await update_garage.ainvoke(
                {"type": "tool_call", "name": "update_garage", "id": "t1",
                 "args": {"wishlist": ["supercharger"]}},
                config={"configurable": {"user_id": user_id}},
            )
            assert "goal" in str(result.content).lower()
            profile = (await client.get(f"/garage/{user_id}")).json()["profile"]
            assert not profile.get("cars"), profile

            # goals passed alongside still merge; still no car
            result = await update_garage.ainvoke(
                {"type": "tool_call", "name": "update_garage", "id": "t2",
                 "args": {"wishlist": ["supercharger"],
                          "goals": ["Planned upgrade: supercharger"]}},
                config={"configurable": {"user_id": user_id}},
            )
            profile = (await client.get(f"/garage/{user_id}")).json()["profile"]
            assert profile.get("goals") == ["Planned upgrade: supercharger"], profile
            assert not profile.get("cars"), profile
