"""The password change endpoint (PUT /settings/password).

The stored hash is PBKDF2 with a RANDOM salt, so two hashes of the
same password are never equal. The endpoint has to VERIFY the
password against the stored hash, not compare two hashes: comparing
them answered 401 for EVERY user, forever, and nobody noticed because
there was no test for it.
"""
import pytest


async def _register(client, email: str, password: str) -> str:
    r = await client.post("/auth/register", json={
        "name": "Ana", "email": email, "password": password,
    })
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_the_password_can_be_changed(client):
    token = await _register(client, "ana@example.com", "vieja-clave-123")

    r = await client.put(
        "/settings/password",
        json={"current_password": "vieja-clave-123", "new_password": "nueva-clave-456"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True

    # The old password no longer logs in. The new one does.
    old = await client.post("/auth/login", json={
        "email": "ana@example.com", "password": "vieja-clave-123"})
    assert old.status_code == 401, old.text
    new = await client.post("/auth/login", json={
        "email": "ana@example.com", "password": "nueva-clave-456"})
    assert new.status_code == 200, new.text


@pytest.mark.asyncio
async def test_the_password_change_refuses_a_wrong_current_password(client):
    token = await _register(client, "ana@example.com", "vieja-clave-123")

    r = await client.put(
        "/settings/password",
        json={"current_password": "no-es-la-clave", "new_password": "nueva-clave-456"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 401, r.text
    # The password did not change.
    login = await client.post("/auth/login", json={
        "email": "ana@example.com", "password": "vieja-clave-123"})
    assert login.status_code == 200, login.text


@pytest.mark.asyncio
async def test_the_password_change_refuses_a_short_new_password(client):
    token = await _register(client, "ana@example.com", "vieja-clave-123")

    r = await client.put(
        "/settings/password",
        json={"current_password": "vieja-clave-123", "new_password": "corta"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400, r.text


@pytest.mark.asyncio
async def test_the_password_change_needs_a_token(client):
    r = await client.put(
        "/settings/password",
        json={"current_password": "vieja-clave-123", "new_password": "nueva-clave-456"},
    )
    assert r.status_code == 401, r.text
