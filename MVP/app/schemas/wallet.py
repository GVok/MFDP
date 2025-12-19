from pydantic import BaseModel, Field


class BalanceOut(BaseModel):
    user_id: int
    balance_rub: int


class TopUpIn(BaseModel):
    amount_rub: int = Field(gt=0)


class TopUpOut(BaseModel):
    transaction_id: int
    new_balance_rub: int
