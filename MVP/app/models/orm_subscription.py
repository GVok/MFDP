from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Boolean, UniqueConstraint, Index
from sqlalchemy.orm import relationship

from db.base import Base
from core.base_classes import BaseEntity


class UserSubscriptionEntity(Base, BaseEntity):
    __tablename__ = "user_subscriptions"

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    plan = Column(String, nullable=False, default="freemium")
    status = Column(String, nullable=False, default="active")

    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False)

    auto_renew = Column(Boolean, nullable=False, default=True)

    user = relationship("UserEntity", back_populates="subscription")

    __table_args__ = (
        UniqueConstraint("user_id", name="uq_user_subscriptions_user_id"),
        Index("ix_user_subscriptions_status", "status"),
    )
