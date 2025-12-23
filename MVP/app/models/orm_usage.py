from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint, Index
from sqlalchemy.orm import relationship

from db.base import Base
from core.base_classes import BaseEntity


class UserMonthlyUsageEntity(Base, BaseEntity):
    __tablename__ = "user_monthly_usage"

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    period_yyyymm = Column(Integer, nullable=False)
    images_generated = Column(Integer, nullable=False, default=0)

    user = relationship("UserEntity", back_populates="usage_rows")

    __table_args__ = (
        UniqueConstraint("user_id", "period_yyyymm", name="uq_user_monthly_usage_user_period"),
        Index("ix_user_monthly_usage_period", "period_yyyymm"),
    )
