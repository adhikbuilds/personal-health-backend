import uuid


def _email() -> str:
    return f"u_{uuid.uuid4().hex[:10]}@example.com"


def test_register_login_me(client):
    email = _email()
    r = client.post("/auth/register", json={"email": email, "name": "Test", "password": "Sup3rsecret!"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert "access_token" in body and "refresh_token" in body
    access = body["access_token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {access}"})
    assert me.status_code == 200
    assert me.json()["email"] == email

    login = client.post("/auth/login", json={"email": email, "password": "Sup3rsecret!"})
    assert login.status_code == 200
    assert login.json()["access_token"]


def test_me_requires_token(anonymous_client):
    r = anonymous_client.get("/auth/me")
    assert r.status_code == 401


def test_register_duplicate(client):
    email = _email()
    client.post("/auth/register", json={"email": email, "name": "A", "password": "Sup3rsecret!"})
    r2 = client.post("/auth/register", json={"email": email, "name": "B", "password": "Sup3rsecret!"})
    assert r2.status_code == 409


def test_register_short_password(client):
    r = client.post("/auth/register", json={"email": _email(), "name": "X", "password": "short"})
    assert r.status_code in (400, 422)


def test_refresh_rotation(client):
    email = _email()
    r = client.post("/auth/register", json={"email": email, "name": "R", "password": "Sup3rsecret!"})
    refresh = r.json()["refresh_token"]
    rot = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert rot.status_code == 200
    new_refresh = rot.json()["refresh_token"]
    # old refresh is now revoked (single-use)
    again = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert again.status_code == 401
    # new refresh works
    rot2 = client.post("/auth/refresh", json={"refresh_token": new_refresh})
    assert rot2.status_code == 200
