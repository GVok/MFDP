from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api.deps import get_db
from web.deps import get_user_from_cookie
from models.orm_user import UserEntity
from services.ml_service import create_ml_request, list_user_requests, get_task, get_request
from services.subscription_service import get_active_plan, get_or_create_month_usage, _yyyymm, _now_utc
from services.wallet_service import get_wallet
from services.brand_profile_service import list_profiles

router = APIRouter()


def _require_user(user: UserEntity | None):
    if not user:
        return RedirectResponse("/login", status_code=302)
    return None

def _img_url(path: str) -> str:
    if not path:
        return path
    if path.startswith("/static/images/"):
        filename = path.split("/static/images/", 1)[1]
        return f"/media/{filename}"
    return path

def _nice_status(s: str) -> str:
    return (s or "").lower().strip()


@router.get("/app")
def app_page(
    request: Request,
    db: Session = Depends(get_db),
    user: UserEntity | None = Depends(get_user_from_cookie),
):
    redir = _require_user(user)
    if redir:
        return redir

    tpl = request.app.state.templates

    plan = get_active_plan(db, user.id)
    wallet = get_wallet(db, user.id)

    ym = _yyyymm(_now_utc())
    usage = get_or_create_month_usage(db, user.id, ym)

    history = list_user_requests(db, user.id, limit=30)

    brand_profiles = []
    if (plan.brand_profiles_limit or 0) > 0:
        brand_profiles = list_profiles(db, user.id)

    return tpl.TemplateResponse(
        "app/index.html",
        {
            "request": request,
            "user": user,
            "plan": plan,
            "wallet": wallet,
            "usage": usage,
            "history": history,
            "brand_profiles": brand_profiles,
            "error": request.query_params.get("err"),
            "ok": request.query_params.get("ok"),
        },
    )


@router.post("/app/generate")
def app_generate(
    request: Request,
    raw_prompt: str = Form(...),
    enhance_backend: str = Form("ollama"),
    image_backend: str = Form("mock"),
    n_images: int = Form(4),
    brand_profile_id: int = Form(0),
    db: Session = Depends(get_db),
    user: UserEntity | None = Depends(get_user_from_cookie),
):
    redir = _require_user(user)
    if redir:
        return redir

    raw_prompt = (raw_prompt or "").strip()
    if not raw_prompt:
        return RedirectResponse("/app?err=empty", status_code=303)

    brand_profile_id = int(brand_profile_id or 0)
    if brand_profile_id <= 0:
        brand_profile_id = None

    try:
        req, task = create_ml_request(
            db,
            user.id,
            raw_prompt,
            enhance_backend=enhance_backend,
            image_backend=image_backend,
            n_images=n_images,
            brand_profile_id=brand_profile_id,
        )
    except ValueError as e:
        return RedirectResponse(f"/app?err={str(e)}", status_code=303)
    except Exception:
        return RedirectResponse("/app?err=server", status_code=303)

    return RedirectResponse(f"/app/tasks/{task.id}", status_code=303)


@router.get("/app/tasks/{task_id}")
def app_task_page(
    task_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: UserEntity | None = Depends(get_user_from_cookie),
):
    redir = _require_user(user)
    if redir:
        return redir

    task = get_task(db, user.id, task_id)
    if not task:
        return RedirectResponse("/app?err=task_not_found", status_code=303)

    tpl = request.app.state.templates

    status = _nice_status(task.status)
    refresh_sec = 2 if status in ("pending", "running") else None

    best_image = None
    req_id = task.request_id

    if task.result and isinstance(task.result, dict):
        best_image = task.result.get("best_image_path")

    if best_image:
        best_image = _img_url(best_image)

    return tpl.TemplateResponse(
        "app/task.html",
        {
            "request": request,
            "user": user,
            "task": task,
            "status": status,
            "refresh_sec": refresh_sec,
            "best_image": best_image,
            "request_id": req_id,
        },
    )


@router.get("/app/requests/{request_id}")
def app_request_page(
    request_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: UserEntity | None = Depends(get_user_from_cookie),
):
    redir = _require_user(user)
    if redir:
        return redir

    req = get_request(db, user.id, request_id)
    if not req:
        return RedirectResponse("/app?err=req_not_found", status_code=303)
    
    preds = list(req.predictions or [])
    for p in preds:
        p.image_path = _img_url(p.image_path)

    tpl = request.app.state.templates
    return tpl.TemplateResponse(
        "app/request.html",
        {
            "request": request,
            "user": user,
            "req": req,
            "predictions": preds,
        },
    )
