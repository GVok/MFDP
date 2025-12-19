from enum import Enum


class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"


class TransactionType(str, Enum):
    TOP_UP = "top_up"
    ML_CHARGE = "ml_charge"
    ADMIN_ADJUST = "admin_adjust"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class ModelType(str, Enum):
    MOCK = "mock"
    STABLE_DIFFUSION = "stable_diffusion"
