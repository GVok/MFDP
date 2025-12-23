from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from db.base import Base
from core.base_classes import BaseEntity


class MLRequestEntity(Base, BaseEntity):
    __tablename__ = "ml_requests"

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    raw_prompt = Column(String, nullable=False)
    cleaned_prompt = Column(String, nullable=True)
    enhanced_prompt = Column(String, nullable=True)

    status = Column(String, nullable=False, default="pending")
    cost_rub = Column(Integer, nullable=False, default=0)

    user = relationship("UserEntity", back_populates="requests")

    tasks = relationship(
        "MLTaskEntity",
        back_populates="request",
        cascade="all, delete-orphan",
    )

    predictions = relationship(
        "PredictionEntity",
        back_populates="request",
        cascade="all, delete-orphan",
        order_by="PredictionEntity.rank",
        )

    transactions = relationship(
        "TransactionEntity",
        back_populates="ml_request",
        foreign_keys="TransactionEntity.ml_request_id",
    )
