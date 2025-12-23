from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api.deps import get_db
from web.deps import get_user_from_cookie
from services.wallet_service import get_wallet, top_up
from services.subscription_service import PLANS, get_active_plan, subscribe
from services.brand_profile_service import list_profiles, create_profile, delete_profile

from models.orm_user import UserEntity

router = APIRouter()

templates = None


PLAN_ORDER = ["freemium", "standard", "business", "pro"]
PLAN_RANK = {code: i for i, code in enumerate(PLAN_ORDER)}


def _require_user(user: UserEntity | None):
    if not user:
        return RedirectResponse("/login", status_code=302)
    return None


@router.get("/account")
def account_page(
    request: Request,
    db: Session = Depends(get_db),
    user: UserEntity | None = Depends(get_user_from_cookie),
):
    redir = _require_user(user)
    if redir:
        return redir

    wallet = get_wallet(db, user.id)
    current = get_active_plan(db, user.id)

    current_rank = PLAN_RANK.get(current.code, 0)

    plans_ui = []
    for code in ["standard", "business", "pro"]:
        p = PLANS[code]
        p_rank = PLAN_RANK[code]

        is_current = (code == current.code)
        is_downgrade_or_same = (p_rank <= current_rank)
        can_afford = (wallet.balance_rub >= p.price_rub)

        action = "buy"
        reason = None

        if is_current:
            action = "current"
        elif is_downgrade_or_same:
            action = "blocked"
            reason = "Already on same or higher plan"
        elif not can_afford:
            action = "no_money"
            reason = "Insufficient balance"

        plans_ui.append(
            {
                "code": code,
                "price_rub": p.price_rub,
                "monthly_images_limit": p.monthly_images_limit,
                "features": {
                    "auto_prompt_enhance": p.auto_prompt_enhance,
                    "auto_best_selection": p.auto_best,
                    "brand_profiles_limit": p.brand_profiles_limit,
                    "lora_per_month": p.lora_per_month,
                },
                "action": action,
                "reason": reason,
            }
        )

    profiles = list_profiles(db, user.id)
    can_manage_brand_profiles = current.brand_profiles_limit > 0
    limit = int(current.brand_profiles_limit or 0)
    used = len(profiles)
    can_create_more = can_manage_brand_profiles and used < limit

    tpl = request.app.state.templates
    return tpl.TemplateResponse(
        "account/dashboard.html",
        {
            "request": request,
            "user": user,
            "balance_rub": wallet.balance_rub,
            "current_plan": current,
            "plans_ui": plans_ui,
            "brand_profiles": profiles,
            "bp_can_manage": can_manage_brand_profiles,
            "bp_limit": limit,
            "bp_used": used,
            "bp_can_create": can_create_more,
            "bp_err": request.query_params.get("bp_err"),
            "bp_ok": request.query_params.get("bp_ok"),
        },
    )


@router.get("/account/topup")
def topup_page(
    request: Request,
    user: UserEntity | None = Depends(get_user_from_cookie),
):
    redir = _require_user(user)
    if redir:
        return redir

    tpl = request.app.state.templates
    return tpl.TemplateResponse(
        "account/topup.html",
        {"request": request, "user": user, "error": None},
    )


@router.post("/account/topup")
def topup_submit(
    request: Request,
    amount_rub: int = Form(...),
    db: Session = Depends(get_db),
    user: UserEntity | None = Depends(get_user_from_cookie),
):
    redir = _require_user(user)
    if redir:
        return redir

    if amount_rub <= 0:
        tpl = request.app.state.templates
        return tpl.TemplateResponse(
            "account/topup.html",
            {"request": request, "user": user, "error": "Amount must be > 0"},
        )

    top_up(db, user.id, amount_rub)
    return RedirectResponse("/account", status_code=302)


@router.post("/account/subscribe")
def subscribe_submit(
    request: Request,
    plan: str = Form(...),
    db: Session = Depends(get_db),
    user: UserEntity | None = Depends(get_user_from_cookie),
):
    redir = _require_user(user)
    if redir:
        return redir

    plan = (plan or "").strip()
    if plan not in PLANS or plan == "freemium":
        return RedirectResponse("/account", status_code=302)

    current = get_active_plan(db, user.id)

    if PLAN_RANK.get(plan, 0) <= PLAN_RANK.get(current.code, 0):
        return RedirectResponse("/account?err=downgrade", status_code=302)

    try:
        subscribe(db, user.id, plan)
        return RedirectResponse("/account?ok=plan", status_code=302)
    except ValueError:
        return RedirectResponse("/account?err=money", status_code=302)

@router.post("/account/brand-profiles/create")
def brand_profile_create(
    request: Request,
    name: str = Form(...),

    mood: str = Form(""),
    category: str = Form(""),
    lighting: str = Form(""),
    color_vibe: str = Form(""),
    composition: str = Form(""),

    materials: list[str] = Form(default=[]),
    avoid: list[str] = Form(default=[]),

    no_color_codes: str | None = Form(default="1"),

    db: Session = Depends(get_db),
    user: UserEntity | None = Depends(get_user_from_cookie),
):
    redir = _require_user(user)
    if redir:
        return redir

    from services.brand_profile_service import build_style_short_from_form

    style_short = build_style_short_from_form(
        mood=mood,
        category=category,
        lighting=lighting,
        color_vibe=color_vibe,
        composition=composition,
        materials=materials,
        avoid=avoid,
        no_color_codes=(no_color_codes is not None),
        max_len=350,
    )

    try:
        create_profile(db, user.id, (name or "").strip(), style_short)
        return RedirectResponse("/account?bp_ok=created", status_code=303)
    except ValueError:
        return RedirectResponse("/account?bp_err=not_allowed", status_code=303)
    except Exception:
        return RedirectResponse("/account?bp_err=server", status_code=303)



@router.post("/account/brand-profiles/{profile_id}/delete")
def brand_profile_delete(
    profile_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: UserEntity | None = Depends(get_user_from_cookie),
):
    redir = _require_user(user)
    if redir:
        return redir

    try:
        ok = delete_profile(db, user.id, profile_id)
        if not ok:
            return RedirectResponse("/account?bp_err=not_found", status_code=303)
        return RedirectResponse("/account?bp_ok=deleted", status_code=303)
    except Exception:
        return RedirectResponse("/account?bp_err=server", status_code=303)
