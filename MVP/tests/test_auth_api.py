def test_register_login_me_happy_path(public_client):

    r = public_client.post(
        "/api/auth/register",
        json={"username": "u01", "email": "u01@example.com", "password": "secret12"},
    )
    assert r.status_code == 201, r.text


    r = public_client.post("/api/auth/login", json={"username": "u01", "password": "secret12"})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    assert token


    r = public_client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    me = r.json()
    assert me["username"] == "u01"
    assert me["email"] == "u01@example.com"
    assert me["is_active"] is True


def test_register_rejects_duplicate_username_or_email(public_client):
    r1 = public_client.post(
        "/api/auth/register",
        json={"username": "dup", "email": "dup@example.com", "password": "secret12"},
    )
    assert r1.status_code == 201, r1.text

    r2 = public_client.post(
        "/api/auth/register",
        json={"username": "dup", "email": "other@example.com", "password": "secret12"},
    )
    assert r2.status_code == 400, r2.text

    r3 = public_client.post(
        "/api/auth/register",
        json={"username": "other", "email": "dup@example.com", "password": "secret12"},
    )
    assert r3.status_code == 400, r3.text


def test_login_wrong_password_returns_401(public_client):
    public_client.post(
        "/api/auth/register",
        json={"username": "u2", "email": "u2@example.com", "password": "secret12"},
    )

    r = public_client.post("/api/auth/login", json={"username": "u2", "password": "WRONG"})
    assert r.status_code == 401, r.text


def test_me_without_token_returns_401(public_client):
    r = public_client.get("/api/auth/me")
    assert r.status_code == 401, r.text
