from fastapi import Cookie, Depends
from sqlalchemy.orm import Session

from api.deps import get_db
from models.orm_user import UserEntity
from services.auth_service import decode_token


def get_user_from_cookie(
    access_token: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
) -> UserEntity | None:
    if not access_token:
        return None

    try:
        payload = decode_token(access_token)
    except Exception:
        return None

    user_id = payload.get("user_id")
    if not user_id:
        return None

    user = (
        db.query(UserEntity)
        .filter(UserEntity.id == user_id, UserEntity.is_active.is_(True))
        .first()
    )
    return user
