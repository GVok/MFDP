import bcrypt

from core.base_classes import IEntity
from models.enums import UserRole
from models.transaction import Transaction
from models.wallet import Wallet


class User(IEntity):
    def __init__(self, user_id: int, username: str, email: str, password: str):
        super().__init__(user_id)

        self._username = username
        self._email = email
        self._password_hash = self._hash_password(password)
        self._role = UserRole.USER
        self._wallet = Wallet(user_id=user_id)

    def _hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def check_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode(), self._password_hash.encode())

    @property
    def balance(self) -> int:
        return self._wallet.balance

    def top_up(self, amount: int, tx_id: int) -> Transaction:
        return self._wallet.deposit(amount, tx_id)

    def charge_for_ml(self, amount: int, tx_id: int) -> Transaction:
        return self._wallet.charge(amount, tx_id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self._username,
            "email": self._email,
            "role": self._role.value,
            "balance_rub": self.balance,
            "created_at": self.created_at.isoformat(),
        }
