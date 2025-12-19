import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    app_debug: bool = os.getenv("APP_DEBUG", "true").lower() == "true"
    secret_key: str = os.getenv("APP_SECRET_KEY", "supersecret")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://ml_user:ml_password@database:5432/ml_service",
    )

    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
    rabbitmq_queue: str = os.getenv("RABBITMQ_QUEUE", "ml_tasks_queue")

    ollama_url: str = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30s")
    ollama_timeout_sec: float = float(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
    enhance_max_tokens: int = int(os.getenv("ENHANCE_MAX_TOKENS", "200"))

    data_dir: str = os.getenv("DATA_DIR", "/data")
    images_dir: str = os.getenv("IMAGES_DIR", "/data/images")

    hf_token: str = os.getenv("HF_TOKEN", "")
    hf_t2i_model: str = os.getenv("HF_T2I_MODEL", "stabilityai/sdxl-turbo")
    hf_t2i_width: int = int(os.getenv("HF_T2I_WIDTH", "512"))
    hf_t2i_height: int = int(os.getenv("HF_T2I_HEIGHT", "512"))
    hf_t2i_steps: int = int(os.getenv("HF_T2I_STEPS", "4"))
    hf_t2i_guidance: float = float(os.getenv("HF_T2I_GUIDANCE", "0.0"))
    hf_t2i_seed: int = int(os.getenv("HF_T2I_SEED", "123"))
    hf_timeout_sec: float = float(os.getenv("HF_TIMEOUT_SEC", "120"))

    image_backend: str = os.getenv("IMAGE_BACKEND", "hf" if os.getenv("HF_TOKEN") else "mock")


settings = Settings()
