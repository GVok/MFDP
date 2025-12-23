import pytest

def test_create_request_happy_path(client, monkeypatch):
    monkeypatch.setattr("services.ml_service.can_generate_images", lambda db, user_id, n: (True, None))

    class Plan:
        auto_prompt_enhance = True
    monkeypatch.setattr("services.ml_service.get_active_plan", lambda db, user_id: Plan())

    published = {"called": False, "task_id": None}
    def _pub(task_id: int):
        published["called"] = True
        published["task_id"] = task_id

    monkeypatch.setattr("services.ml_service.publish_task", _pub)

    r = client.post("/api/ml/requests", json={"raw_prompt": "котик", "enhance_backend": "ollama", "image_backend": "mock", "n_images": 3})
    assert r.status_code == 201, r.text
    data = r.json()
    assert "request_id" in data
    assert "task_id" in data
    assert data["status"] == "pending"
    assert published["called"] is True
    assert published["task_id"] == data["task_id"]


@pytest.mark.parametrize("payload", [
    {"raw_prompt": "test", "enhance_backend": "ollama", "image_backend": "mock", "n_images": 0},
    {"raw_prompt": "test", "enhance_backend": "ollama", "image_backend": "mock", "n_images": 999},
    {"raw_prompt": "test", "enhance_backend": "ollama", "image_backend": "local", "n_images": 1},
])
def test_create_request_rejects_invalid_payload(client, monkeypatch, payload):
    monkeypatch.setattr("services.ml_service.can_generate_images", lambda db, user_id, n: (True, None))
    class Plan:
        auto_prompt_enhance = True
    monkeypatch.setattr("services.ml_service.get_active_plan", lambda db, user_id: Plan())
    monkeypatch.setattr("services.ml_service.publish_task", lambda task_id: None)

    r = client.post("/api/ml/requests", json=payload)
    assert r.status_code == 422, r.text


def test_plan_disables_auto_enhance(client, monkeypatch):
    monkeypatch.setattr("services.ml_service.can_generate_images", lambda db, user_id, n: (True, None))
    class Plan:
        auto_prompt_enhance = False
    monkeypatch.setattr("services.ml_service.get_active_plan", lambda db, user_id: Plan())
    monkeypatch.setattr("services.ml_service.publish_task", lambda task_id: None)

    r = client.post("/api/ml/requests", json={"raw_prompt": "test", "enhance_backend": "ollama", "image_backend": "mock", "n_images": 1})
    assert r.status_code == 201
    task_id = r.json()["task_id"]

    t = client.get(f"/api/ml/tasks/{task_id}")
    payload = t.json()["payload"]
    assert payload["enhance_backend"] == "mock"


def test_monthly_limit_returns_402(client, monkeypatch):
    monkeypatch.setattr("services.ml_service.can_generate_images", lambda db, user_id, n: (False, "limit"))
    monkeypatch.setattr("services.ml_service.publish_task", lambda task_id: (_ for _ in ()).throw(AssertionError("should not publish")))

    r = client.post("/api/ml/requests", json={"raw_prompt": "test", "enhance_backend": "ollama", "image_backend": "mock", "n_images": 4})
    assert r.status_code == 402
    assert "limit" in r.text.lower()
