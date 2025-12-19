from sqlalchemy import Column, Float, ForeignKey, Integer, String, JSON
from sqlalchemy.orm import relationship

from db.base import Base
from core.base_classes import BaseEntity


class PredictionEntity(Base, BaseEntity):
    __tablename__ = "predictions"

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

    image_path = Column(String, nullable=False)

    clip_score = Column(Float, nullable=True)
    aesthetic_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True)
    rank = Column(Integer, nullable=True)

    gen_meta = Column(JSON, nullable=False, default=dict)

    user = relationship("UserEntity", back_populates="predictions")
    request = relationship("MLRequestEntity", back_populates="predictions")
