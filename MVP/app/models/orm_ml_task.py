from sqlalchemy import Column, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from db.base import Base
from core.base_classes import BaseEntity


class MLTaskEntity(Base, BaseEntity):
    __tablename__ = "ml_tasks"

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    request_id = Column(
        Integer,
        ForeignKey("ml_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status = Column(String, nullable=False, default="pending")
    payload = Column(JSON, nullable=False, default=dict)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)

    user = relationship("UserEntity", back_populates="tasks")
    request = relationship("MLRequestEntity", back_populates="tasks")
