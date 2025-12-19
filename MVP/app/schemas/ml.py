from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any, Dict


class CreateMLRequestIn(BaseModel):
    raw_prompt: str = Field(min_length=1, max_length=2000)

    enhance_backend: Literal["ollama", "mock"] = "ollama"
    image_backend: Literal["hf", "mock"] = "mock"

    n_images: int = Field(default=1, ge=1, le=4)


class CreateMLRequestOut(BaseModel):
    request_id: int
    task_id: int
    status: str


class MLRequestOut(BaseModel):
    id: int
    raw_prompt: str
    cleaned_prompt: Optional[str]
    enhanced_prompt: Optional[str]
    status: str
    cost_rub: int

    class Config:
        from_attributes = True


class MLHistoryOut(BaseModel):
    requests: List[MLRequestOut]


class MLTaskOut(BaseModel):
    id: int
    request_id: int
    status: str
    result: Optional[Dict[str, Any]]
    error: Optional[str]

    class Config:
        from_attributes = True
