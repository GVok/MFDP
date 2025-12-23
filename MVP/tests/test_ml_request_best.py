from models.orm_ml_request import MLRequestEntity
from models.orm_prediction import PredictionEntity
from models.enums import TaskStatus

def test_best_prediction_prefers_rank(client, db_session, monkeypatch):
    monkeypatch.setattr("services.ml_service.can_generate_images", lambda db, user_id, n: (True, None))
    class Plan:
        auto_prompt_enhance = True
    monkeypatch.setattr("services.ml_service.get_active_plan", lambda db, user_id: Plan())
    monkeypatch.setattr("services.ml_service.publish_task", lambda task_id: None)

    r = client.post("/api/ml/requests", json={"raw_prompt": "test", "enhance_backend": "ollama", "image_backend": "mock", "n_images": 1})
    req_id = r.json()["request_id"]

    p1 = PredictionEntity(user_id=1, request_id=req_id, image_path="/a.png", rank=2, final_score=0.9, clip_score=0.1, aesthetic_score=0.1)
    p2 = PredictionEntity(user_id=1, request_id=req_id, image_path="/b.png", rank=1, final_score=0.2, clip_score=0.1, aesthetic_score=0.1)
    db_session.add_all([p1, p2])
    db_session.commit()

    out = client.get(f"/api/ml/requests/{req_id}").json()
    assert out["best"]["image_path"] == "/b.png"


def test_best_prediction_fallbacks_to_final_score(client, db_session, monkeypatch):
    monkeypatch.setattr("services.ml_service.can_generate_images", lambda db, user_id, n: (True, None))
    class Plan:
        auto_prompt_enhance = True
    monkeypatch.setattr("services.ml_service.get_active_plan", lambda db, user_id: Plan())
    monkeypatch.setattr("services.ml_service.publish_task", lambda task_id: None)

    r = client.post(f"/api/ml/requests", json={"raw_prompt": "test2", "enhance_backend": "ollama", "image_backend": "mock", "n_images": 1})
    req_id = r.json()["request_id"]

    p1 = PredictionEntity(user_id=1, request_id=req_id, image_path="/a.png", rank=None, final_score=0.3, clip_score=0.1, aesthetic_score=0.1)
    p2 = PredictionEntity(user_id=1, request_id=req_id, image_path="/b.png", rank=None, final_score=0.8, clip_score=0.1, aesthetic_score=0.1)
    db_session.add_all([p1, p2])
    db_session.commit()

    out = client.get(f"/api/ml/requests/{req_id}").json()
    assert out["best"]["image_path"] == "/b.png"
