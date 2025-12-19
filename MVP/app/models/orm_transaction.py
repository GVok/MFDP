from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from db.base import Base
from core.base_classes import BaseEntity


class TransactionEntity(Base, BaseEntity):
    __tablename__ = "transactions"

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    amount_rub = Column(Integer, nullable=False)
    type = Column(String, nullable=False)
    description = Column(String, nullable=True)

    ml_request_id = Column(
        Integer,
        ForeignKey("ml_requests.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_by_admin_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    user = relationship(
        "UserEntity",
        back_populates="transactions",
        foreign_keys=[user_id],
    )

    ml_request = relationship(
        "MLRequestEntity",
        back_populates="transactions",
        foreign_keys=[ml_request_id],
    )

    created_by_admin = relationship(
        "UserEntity",
        foreign_keys=[created_by_admin_id],
    )
