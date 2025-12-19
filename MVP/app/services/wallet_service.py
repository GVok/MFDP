from sqlalchemy.orm import Session

from models.orm_wallet import WalletEntity
from models.orm_transaction import TransactionEntity
from models.enums import TransactionType


def get_wallet(db: Session, user_id: int) -> WalletEntity:
    wallet = db.query(WalletEntity).filter(WalletEntity.user_id == user_id).first()
    if not wallet:
        wallet = WalletEntity(user_id=user_id, balance_rub=0)
        db.add(wallet)
        db.commit()
        db.refresh(wallet)
    return wallet


def top_up(db: Session, user_id: int, amount_rub: int) -> tuple[int, int]:
    wallet = get_wallet(db, user_id)
    wallet.balance_rub += amount_rub

    tx = TransactionEntity(
        user_id=user_id,
        amount_rub=amount_rub,
        type=TransactionType.TOP_UP.value,
        description="User top-up",
        ml_request_id=None,
        created_by_admin_id=None,
    )
    db.add(tx)
    db.commit()
    db.refresh(tx)
    db.refresh(wallet)
    return tx.id, wallet.balance_rub
