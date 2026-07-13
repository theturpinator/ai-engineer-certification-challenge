"""Seam-1: the raw-bytes photo upload path (issue #53) — real Postgres, no
LLM. Uploading stores the image verbatim with its content type, and
re-uploading replaces it (the Change photo / Rotate 90° affordances both
ride this PUT)."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from psycopg.types.json import Json

import app as app_module
from app import app


@pytest.mark.asyncio
async def test_upload_then_reupload_replaces_the_stored_image():
    user_id = str(uuid.uuid4())
    async with app.router.lifespan_context(app):
        async with await app_module._db() as conn:
            await conn.execute(
                "INSERT INTO garage (user_id, profile) VALUES (%s, %s)",
                (user_id, Json({"cars": [{"id": "car1", "year": 2016,
                                          "trim": "GT"}]})),
            )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = b"\xff\xd8\xff first-upload-bytes"
            r = await client.put(
                f"/garage/{user_id}/cars/car1/image",
                content=first, headers={"Content-Type": "image/jpeg"},
            )
            assert r.status_code == 204

            r = await client.get(f"/garage/{user_id}/cars/car1/image")
            assert r.status_code == 200
            assert r.content == first
            assert r.headers["content-type"].startswith("image/jpeg")

            # re-upload (Change photo / Rotate 90°) replaces, same endpoint
            second = b"\x89PNG replacement-bytes"
            r = await client.put(
                f"/garage/{user_id}/cars/car1/image",
                content=second, headers={"Content-Type": "image/png"},
            )
            assert r.status_code == 204

            r = await client.get(f"/garage/{user_id}/cars/car1/image")
            assert r.content == second
            assert r.headers["content-type"].startswith("image/png")
