from models.user import User
from models.enums import UserRole


class Admin(User):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._role = UserRole.ADMIN

    def adjust_balance(self, user: User, amount: int, tx_id: int):
        if amount > 0:
            return user.top_up(amount, tx_id)
        return user.charge_for_ml(abs(amount), tx_id)
