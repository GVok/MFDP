from sqlalchemy import Boolean, Column, String
from sqlalchemy.orm import relationship

from db.base import Base
from core.base_classes import BaseEntity


class UserEntity(Base, BaseEntity):
    __tablename__ = "users"

    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)

    is_admin = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    wallet = relationship(
        "WalletEntity",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )

    transactions = relationship(
        "TransactionEntity",
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="TransactionEntity.user_id",
    )

    tasks = relationship(
        "MLTaskEntity",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    predictions = relationship(
        "PredictionEntity",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    requests = relationship(
        "MLRequestEntity",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    subscription = relationship(
        "UserSubscriptionEntity",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )

    usage_rows = relationship(
        "UserMonthlyUsageEntity",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    brand_profiles = relationship(
            "BrandProfileEntity",
            back_populates="user",
            cascade="all, delete-orphan",
    )
