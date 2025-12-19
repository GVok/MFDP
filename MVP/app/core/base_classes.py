from abc import ABC, abstractmethod
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, func
from sqlalchemy.orm import declared_attr


class IEntity(ABC):
    def __init__(self, entity_id: int):
        self._id = entity_id
        self._created_at = datetime.utcnow()

    @property
    def id(self) -> int:
        return self._id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class BaseEntity:
    __abstract__ = True

    @declared_attr
    def id(cls):
        return Column(Integer, primary_key=True, index=True)

    @declared_attr
    def created_at(cls):
        return Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
