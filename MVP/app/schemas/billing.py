from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal

PlanCode = Literal["freemium", "standard", "business", "pro"]


class PlanOut(BaseModel):
    code: PlanCode
    price_rub: int
    monthly_images_limit: Optional[int]
    features: Dict[str, Any]


class PlansOut(BaseModel):
    plans: List[PlanOut]


class MyBillingOut(BaseModel):
    plan: PlanOut
    period_yyyymm: int
    images_generated: int


class SubscribeIn(BaseModel):
    plan: PlanCode


class SubscribeOut(BaseModel):
    status: str
    plan: PlanOut
    current_period_start: str
    current_period_end: str
