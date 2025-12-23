from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint, Index
from sqlalchemy.orm import relationship

from db.base import Base
from core.base_classes import BaseEntity


class BrandProfileEntity(Base, BaseEntity):
    __tablename__ = "brand_profiles"

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name = Column(String(80), nullable=False)
    style_short = Column(String(2000), nullable=False, default="")

    user = relationship("UserEntity")

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_brand_profiles_user_name"),
        Index("ix_brand_profiles_user_id", "user_id"),
    )
