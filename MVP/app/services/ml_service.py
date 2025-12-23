from sqlalchemy.orm import Session
from sqlalchemy.orm import selectinload

from models.orm_ml_request import MLRequestEntity
from models.orm_ml_task import MLTaskEntity
from models.orm_brand_profile import BrandProfileEntity
from models.enums import TaskStatus
from services.rabbitmq import publish_task
from services.subscription_service import can_generate_images, get_active_plan

DEFAULT_COST_RUB = 0


def _resolve_brand_style_short(
    db: Session,
    user_id: int,
    brand_profile_id: int | None,
) -> tuple[str, int | None, str | None]:
    """
    Возвращает (brand_style_short, brand_profile_id, brand_profile_name)

    По требованию: если НЕ доступно / не найдено / не его -> "" (пустая строка).
    """
    plan = get_active_plan(db, user_id)

    if plan.brand_profiles_limit <= 0:
        return "", None, None

    if not brand_profile_id:
        return "", None, None

    prof = (
        db.query(BrandProfileEntity)
        .filter(BrandProfileEntity.user_id == user_id, BrandProfileEntity.id == int(brand_profile_id))
        .first()
    )
    if not prof:
        return "", None, None

    return (prof.style_short or ""), prof.id, prof.name


def create_ml_request(
    db: Session,
    user_id: int,
    raw_prompt: str,
    *,
    enhance_backend: str = "ollama",
    image_backend: str = "mock",
    n_images: int = 1,
    brand_profile_id: int | None = None,
) -> tuple[MLRequestEntity, MLTaskEntity]:
    if image_backend not in ("mock", "local"):
        image_backend = "mock"

    if image_backend == "local":
        n_images_eff = max(1, min(int(n_images or 1), 2))
    else:
        n_images_eff = max(1, min(int(n_images or 4), 4))

    ok, reason = can_generate_images(db, user_id, n_images_eff)
    if not ok:
        raise ValueError(reason or "Monthly limit reached")

    plan = get_active_plan(db, user_id)
    if not plan.auto_prompt_enhance:
        enhance_backend = "mock"

    brand_style_short, used_profile_id, used_profile_name = _resolve_brand_style_short(
        db, user_id, brand_profile_id
    )

    req = MLRequestEntity(
        user_id=user_id,
        raw_prompt=raw_prompt,
        cleaned_prompt=None,
        enhanced_prompt=None,
        status=TaskStatus.PENDING.value,
        cost_rub=DEFAULT_COST_RUB,
    )
    db.add(req)
    db.commit()
    db.refresh(req)

    payload = {
        "raw_prompt": raw_prompt,
        "enhance_backend": enhance_backend,
        "image_backend": image_backend,
        "n_images": n_images_eff,
        "brand_style_short": brand_style_short,
        "brand_profile_id": used_profile_id,
        "brand_profile_name": used_profile_name,
    }

    task = MLTaskEntity(
        user_id=user_id,
        request_id=req.id,
        status=TaskStatus.PENDING.value,
        payload=payload,
        result=None,
        error=None,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    publish_task(task.id)
    return req, task


def list_user_requests(db: Session, user_id: int, limit: int = 50) -> list[MLRequestEntity]:
    return (
        db.query(MLRequestEntity)
        .filter(MLRequestEntity.user_id == user_id)
        .order_by(MLRequestEntity.id.desc())
        .limit(limit)
        .all()
    )


def get_request(db: Session, user_id: int, request_id: int) -> MLRequestEntity | None:
    return (
        db.query(MLRequestEntity)
        .options(selectinload(MLRequestEntity.predictions))
        .filter(MLRequestEntity.user_id == user_id, MLRequestEntity.id == request_id)
        .first()
    )


def get_task(db: Session, user_id: int, task_id: int) -> MLTaskEntity | None:
    return (
        db.query(MLTaskEntity)
        .filter(MLTaskEntity.id == task_id, MLTaskEntity.user_id == user_id)
        .first()
    )
