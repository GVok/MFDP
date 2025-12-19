from core.base_classes import IEntity
from models.enums import TransactionType


class Transaction(IEntity):
    def __init__(
        self,
        transaction_id: int,
        user_id: int,
        amount_rub: int,
        tx_type: TransactionType,
        description: str | None = None,
    ):
        super().__init__(transaction_id)

        if amount_rub == 0:
            raise ValueError("Transaction amount cannot be zero")

        self._user_id = user_id
        self._amount_rub = amount_rub
        self._type = tx_type
        self._description = description

    @property
    def amount(self) -> int:
        return self._amount_rub

    @property
    def type(self) -> TransactionType:
        return self._type

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self._user_id,
            "amount_rub": self._amount_rub,
            "type": self._type.value,
            "description": self._description,
            "created_at": self.created_at.isoformat(),
        }
