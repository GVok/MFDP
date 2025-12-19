from models.transaction import Transaction
from models.enums import TransactionType


class Wallet:
    def __init__(self, user_id: int):
        self._user_id = user_id
        self._balance_rub = 0
        self._transactions: list[Transaction] = []

    @property
    def balance(self) -> int:
        return self._balance_rub

    def deposit(self, amount: int, tx_id: int) -> Transaction:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        self._balance_rub += amount
        tx = Transaction(
            transaction_id=tx_id,
            user_id=self._user_id,
            amount_rub=amount,
            tx_type=TransactionType.TOP_UP,
            description="User top-up",
        )
        self._transactions.append(tx)
        return tx

    def charge(self, amount: int, tx_id: int) -> Transaction:
        if amount <= 0:
            raise ValueError("Charge amount must be positive")
        if self._balance_rub < amount:
            raise ValueError("Insufficient balance")

        self._balance_rub -= amount
        tx = Transaction(
            transaction_id=tx_id,
            user_id=self._user_id,
            amount_rub=-amount,
            tx_type=TransactionType.ML_CHARGE,
            description="ML request charge",
        )
        self._transactions.append(tx)
        return tx

    def history(self) -> list[Transaction]:
        return list(self._transactions)
