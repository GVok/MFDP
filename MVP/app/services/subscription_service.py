from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from models.orm_subscription import UserSubscriptionEntity
from models.orm_usage import UserMonthlyUsageEntity
from models.orm_transaction import TransactionEntity
from models.enums import TransactionType
from services.wallet_service import get_wallet


@dataclass(frozen=True)
class Plan:
    code: str
    price_rub: int
    monthly_images_limit: int | None
    auto_prompt_enhance: bool
    auto_best: bool
    brand_profiles_limit: int
    lora_per_month: int


PLANS: dict[str, Plan] = {
    "freemium": Plan("freemium", 0, 10, False, False, 0, 0),
    "standard": Plan("standard", 590, None, True, False, 0, 0),
    "business": Plan("business", 1590, None, True, True, 3, 0),
    "pro": Plan("pro", 3990, None, True, True, 10, 1),
}


def _now_utc() -> datetime:
    return datetime.utcnow()


def _yyyymm(dt: datetime) -> int:
    return dt.year * 100 + dt.month


def get_active_plan(db: Session, user_id: int) -> Plan:
    sub = db.query(UserSubscriptionEntity).filter(UserSubscriptionEntity.user_id == user_id).first()
    if not sub:
        return PLANS["freemium"]

    now = _now_utc()
    if sub.status != "active" or sub.current_period_end <= now:
        return PLANS["freemium"]

    return PLANS.get(sub.plan, PLANS["freemium"])


def get_or_create_month_usage(db: Session, user_id: int, yyyymm: int) -> UserMonthlyUsageEntity:
    row = (
        db.query(UserMonthlyUsageEntity)
        .filter(UserMonthlyUsageEntity.user_id == user_id, UserMonthlyUsageEntity.period_yyyymm == yyyymm)
        .first()
    )
    if row:
        return row

    row = UserMonthlyUsageEntity(user_id=user_id, period_yyyymm=yyyymm, images_generated=0)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def can_generate_images(db: Session, user_id: int, n_images: int) -> tuple[bool, str | None]:
    plan = get_active_plan(db, user_id)
    if plan.monthly_images_limit is None:
        return True, None

    ym = _yyyymm(_now_utc())
    usage = get_or_create_month_usage(db, user_id, ym)
    if usage.images_generated + n_images > plan.monthly_images_limit:
        return False, f"Freemium limit reached: {usage.images_generated}/{plan.monthly_images_limit} images this month"
    return True, None


def add_usage_images(db: Session, user_id: int, n_images: int) -> None:
    ym = _yyyymm(_now_utc())
    usage = get_or_create_month_usage(db, user_id, ym)
    usage.images_generated += int(n_images)
    db.commit()


def subscribe(db: Session, user_id: int, plan_code: str) -> UserSubscriptionEntity:
    if plan_code not in PLANS or plan_code == "freemium":
        raise ValueError("Invalid plan")

    plan = PLANS[plan_code]
    wallet = get_wallet(db, user_id)
    if wallet.balance_rub < plan.price_rub:
        raise ValueError("Insufficient balance")

    now = _now_utc()
    period_start = now
    period_end = now + timedelta(days=30)

    sub = db.query(UserSubscriptionEntity).filter(UserSubscriptionEntity.user_id == user_id).first()
    if not sub:
        sub = UserSubscriptionEntity(
            user_id=user_id,
            plan=plan.code,
            status="active",
            current_period_start=period_start,
            current_period_end=period_end,
            auto_renew=True,
        )
        db.add(sub)
    else:
        sub.plan = plan.code
        sub.status = "active"
        sub.current_period_start = period_start
        sub.current_period_end = period_end
        sub.auto_renew = True

    wallet.balance_rub -= plan.price_rub

    tx = TransactionEntity(
        user_id=user_id,
        amount_rub=-plan.price_rub,
        type=TransactionType.SUBSCRIPTION.value,
        description=f"Subscription purchase: {plan.code}",
        ml_request_id=None,
        created_by_admin_id=None,
    )
    db.add(tx)

    db.commit()
    db.refresh(sub)
    return sub
