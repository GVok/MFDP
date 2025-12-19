from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps import get_db, require_admin
from models.orm_user import UserEntity
from services.wallet_service import top_up


router = APIRouter(prefix="/admin", tags=["Admin"])


class AdminTopUpIn(BaseModel):
    user_id: int
    amount_rub: int = Field(gt=0)


@router.post("/top-up", status_code=201)
def admin_topup(data: AdminTopUpIn, db: Session = Depends(get_db), _: UserEntity = Depends(require_admin)):
    user = db.query(UserEntity).filter(UserEntity.id == data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    tx_id, new_balance = top_up(db, data.user_id, data.amount_rub)
    return {"transaction_id": tx_id, "new_balance_rub": new_balance}
