from sqlalchemy.orm import Session

from models.orm_ml_request import MLRequestEntity
from models.orm_ml_task import MLTaskEntity
from models.enums import TaskStatus
from services.wallet_service import get_wallet
from services.rabbitmq import publish_task



DEFAULT_COST_RUB = 10


def create_ml_request(
    db: Session,
    user_id: int,
    raw_prompt: str,
    *,
    enhance_backend: str = "ollama",
    image_backend: str = "mock",
    n_images: int = 1,
) -> tuple[MLRequestEntity, MLTaskEntity]:
    wallet = get_wallet(db, user_id)
    if wallet.balance_rub < DEFAULT_COST_RUB:
        raise ValueError("Insufficient balance")

    req = MLRequestEntity(
        user_id=user_id,
        raw_prompt=raw_prompt,
        cleaned_prompt=None,
        enhanced_prompt=None,
        status="pending",
        cost_rub=DEFAULT_COST_RUB,
    )
    db.add(req)
    db.commit()
    db.refresh(req)

    if image_backend == "hf":
        n_images_eff = 1
    else:
        n_images_eff = max(1, min(int(n_images or 4), 4))

    payload = {
        "raw_prompt": raw_prompt,
        "enhance_backend": enhance_backend,
        "image_backend": image_backend,
        "n_images": n_images_eff,
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
        .filter(MLRequestEntity.user_id == user_id, MLRequestEntity.id == request_id)
        .first()
    )


def get_task(db: Session, user_id: int, task_id: int) -> MLTaskEntity | None:
    return (
        db.query(MLTaskEntity)
        .filter(MLTaskEntity.id == task_id, MLTaskEntity.user_id == user_id)
        .first()
    )
