from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from models.orm_user import UserEntity
from schemas.billing import PlansOut, PlanOut, MyBillingOut, SubscribeIn, SubscribeOut
from services.subscription_service import PLANS, get_active_plan, get_or_create_month_usage, subscribe, _yyyymm, _now_utc

router = APIRouter(prefix="/billing", tags=["Billing"])


def _plan_out(code: str) -> PlanOut:
    p = PLANS[code]
    return PlanOut(
        code=p.code,
        price_rub=p.price_rub,
        monthly_images_limit=p.monthly_images_limit,
        features={
            "auto_prompt_enhance": p.auto_prompt_enhance,
            "auto_best_selection": p.auto_best,
            "brand_profiles_limit": p.brand_profiles_limit,
            "lora_per_month": p.lora_per_month,
        },
    )


@router.get("/plans", response_model=PlansOut)
def plans():
    return PlansOut(plans=[_plan_out(k) for k in ["freemium", "standard", "business", "pro"]])


@router.get("/me", response_model=MyBillingOut)
def me(user: UserEntity = Depends(get_current_user), db: Session = Depends(get_db)):
    plan = get_active_plan(db, user.id)
    ym = _yyyymm(_now_utc())
    usage = get_or_create_month_usage(db, user.id, ym)
    return MyBillingOut(plan=_plan_out(plan.code), period_yyyymm=ym, images_generated=usage.images_generated)


@router.post("/subscribe", response_model=SubscribeOut, status_code=201)
def do_subscribe(data: SubscribeIn, user: UserEntity = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        sub = subscribe(db, user.id, data.plan)
        p = _plan_out(sub.plan)
        return SubscribeOut(
            status="ok",
            plan=p,
            current_period_start=sub.current_period_start.isoformat(),
            current_period_end=sub.current_period_end.isoformat(),
        )
    except ValueError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to subscribe: {e}")
