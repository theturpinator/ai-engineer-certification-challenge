"""Seam-1: optional Google login — real Postgres, real app JWTs; ONLY the
Google ID-token verification (app._verify_google_token, the JWKS boundary) is
monkeypatched. Covers: forged Google token -> 401; first login binds
google_sub -> the browser's anonymous user_id; a later login from a DIFFERENT
device (different anon id) resolves to the ORIGINAL canonical user_id
(cross-device association); a bearer token overrides the path user id on
user-scoped routes; expired/forged app JWTs -> 401; unconfigured -> 503.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import jwt
import psycopg
import pytest
from httpx import ASGITransport, AsyncClient

import app as app_module
from app import app

SECRET = "test-jwt-secret-0123456789abcdef0123456789abcdef"


def _claims(sub):
    return {"sub": sub, "email": "sam@example.com", "name": "Mustang Sam",
            "picture": "https://example.com/sam.png"}


def _configure(monkeypatch, sub):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AUTH_JWT_SECRET", SECRET)
    monkeypatch.setattr(app_module, "_verify_google_token", lambda tok: _claims(sub))


async def _login(client, anon_user_id):
    resp = await client.post(
        "/auth/google", json={"id_token": "stub", "anon_user_id": anon_user_id}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.asyncio
async def test_google_login_binding_and_cross_device_resolution(monkeypatch):
    sub = f"google-sub-{uuid.uuid4()}"
    device_a, device_b = str(uuid.uuid4()), str(uuid.uuid4())
    _configure(monkeypatch, sub)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # first-ever login binds the anonymous id as canonical (adoption in place)
            first = await _login(client, device_a)
            assert first["user_id"] == device_a
            assert first["name"] == "Mustang Sam" and first["token"]
            with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
                row = conn.execute(
                    "SELECT user_id, email FROM identities WHERE google_sub = %s",
                    (sub,),
                ).fetchone()
            assert row == (device_a, "sam@example.com")

            # same Google account on a fresh device: the canonical id comes back
            second = await _login(client, device_b)
            assert second["user_id"] == device_a  # THE cross-device criterion

            # /auth/me restores the session from the stored app JWT
            me = await client.get(
                "/auth/me", headers={"Authorization": f"Bearer {second['token']}"}
            )
            assert me.status_code == 200
            assert me.json()["user_id"] == device_a
            assert (await client.get("/auth/me")).status_code == 401

            # anon_user_id is validated at the trust boundary
            bad = await client.post(
                "/auth/google", json={"id_token": "stub", "anon_user_id": ""}
            )
            assert bad.status_code == 422


@pytest.mark.asyncio
async def test_bearer_token_overrides_path_user_id(monkeypatch):
    sub = f"google-sub-{uuid.uuid4()}"
    canonical, other_device = str(uuid.uuid4()), str(uuid.uuid4())
    _configure(monkeypatch, sub)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            token = (await _login(client, canonical))["token"]
            with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
                conn.execute(
                    "INSERT INTO chats (user_id, chat_id, title) VALUES (%s, %s, %s)",
                    (canonical, "seeded", "Seeded chat"),
                )
                conn.execute(
                    "INSERT INTO garage (user_id, profile) VALUES (%s, %s)",
                    (canonical, '{"goals": ["track"]}'),
                )
            bearer = {"Authorization": f"Bearer {token}"}

            # signed-in requests land on the canonical id, whatever the path says
            chats = (await client.get(f"/chats/{other_device}", headers=bearer)).json()
            assert [c["chat_id"] for c in chats] == ["seeded"]
            garage = (await client.get(f"/garage/{other_device}", headers=bearer)).json()
            assert garage["profile"]["goals"] == ["track"]

            # no header: anonymous behavior unchanged (that path id is empty)
            assert (await client.get(f"/chats/{other_device}")).json() == []


@pytest.mark.asyncio
async def test_bad_tokens_and_unconfigured(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AUTH_JWT_SECRET", SECRET)

    def forged(tok):
        raise ValueError("Token has wrong audience")

    monkeypatch.setattr(app_module, "_verify_google_token", forged)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # forged/invalid Google ID token
            resp = await client.post(
                "/auth/google", json={"id_token": "forged", "anon_user_id": "a"}
            )
            assert resp.status_code == 401

            # expired app JWT -> explicit 401, not a silent fallback
            expired = jwt.encode(
                {"sub": "s", "uid": "u",
                 "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
                SECRET, algorithm="HS256",
            )
            resp = await client.get(
                "/chats/whoever", headers={"Authorization": f"Bearer {expired}"}
            )
            assert resp.status_code == 401

            # app JWT signed with the wrong secret -> 401
            wrong = jwt.encode({"uid": "u"}, "not-the-secret-" + "x" * 32,
                               algorithm="HS256")
            resp = await client.get(
                f"/garage/{uuid.uuid4()}", headers={"Authorization": f"Bearer {wrong}"}
            )
            assert resp.status_code == 401

            # login disabled until the owner configures a Google client id
            monkeypatch.delenv("GOOGLE_CLIENT_ID")
            resp = await client.post(
                "/auth/google", json={"id_token": "stub", "anon_user_id": "a"}
            )
            assert resp.status_code == 503
