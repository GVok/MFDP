from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from schemas.wallet import BalanceOut, TopUpIn, TopUpOut
from models.orm_user import UserEntity
from services.wallet_service import get_wallet, top_up


router = APIRouter(prefix="/wallet", tags=["Wallet"])


@router.get("/balance", response_model=BalanceOut)
def balance(user: UserEntity = Depends(get_current_user), db: Session = Depends(get_db)):
    wallet = get_wallet(db, user.id)
    return BalanceOut(user_id=user.id, balance_rub=wallet.balance_rub)


@router.post("/top-up", response_model=TopUpOut, status_code=201)
def topup(data: TopUpIn, user: UserEntity = Depends(get_current_user), db: Session = Depends(get_db)):
    tx_id, new_balance = top_up(db, user.id, data.amount_rub)
    return TopUpOut(transaction_id=tx_id, new_balance_rub=new_balance)
