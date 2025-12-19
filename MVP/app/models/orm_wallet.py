from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from db.base import Base
from core.base_classes import BaseEntity


class WalletEntity(Base, BaseEntity):
    __tablename__ = "wallets"

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    balance_rub = Column(Integer, nullable=False, default=0)

    user = relationship("UserEntity", back_populates="wallet")
