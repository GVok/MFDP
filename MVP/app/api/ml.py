from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from schemas.ml import (
    CreateMLRequestIn,
    CreateMLRequestOut,
    MLRequestOut,
    MLHistoryOut,
    MLTaskOut,
    PredictionOut,
)
from models.orm_user import UserEntity
from services.ml_service import create_ml_request, list_user_requests, get_request, get_task

router = APIRouter(prefix="/ml", tags=["ML"])


@router.post("/requests", response_model=CreateMLRequestOut, status_code=201)
def create_request(
    data: CreateMLRequestIn,
    user: UserEntity = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        req, task = create_ml_request(
            db,
            user.id,
            data.raw_prompt,
            enhance_backend=data.enhance_backend,
            image_backend=data.image_backend,
            n_images=data.n_images,
            brand_profile_id=data.brand_profile_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create request: {e}")

    return CreateMLRequestOut(request_id=req.id, task_id=task.id, status=task.status)


@router.get("/requests/{request_id}", response_model=MLRequestOut)
def read_request(request_id: int, user: UserEntity = Depends(get_current_user), db: Session = Depends(get_db)):
    req = get_request(db, user.id, request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")

    preds = list(req.predictions or [])
    best_pred = None

    ranked = [p for p in preds if p.rank is not None]
    if ranked:
        best_pred = sorted(ranked, key=lambda p: p.rank)[0]
    else:
        scored = [p for p in preds if p.final_score is not None]
        if scored:
            best_pred = sorted(scored, key=lambda p: p.final_score, reverse=True)[0]

    out = MLRequestOut.model_validate(req)
    out.best = PredictionOut.model_validate(best_pred) if best_pred else None
    return out


@router.get("/history", response_model=MLHistoryOut)
def history(user: UserEntity = Depends(get_current_user), db: Session = Depends(get_db)):
    reqs = list_user_requests(db, user.id, limit=50)
    return MLHistoryOut(requests=reqs)


@router.get("/tasks/{task_id}", response_model=MLTaskOut)
def read_task(task_id: int, user: UserEntity = Depends(get_current_user), db: Session = Depends(get_db)):
    task = get_task(db, user.id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
