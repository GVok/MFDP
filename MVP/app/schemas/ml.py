from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any, Dict


class CreateMLRequestIn(BaseModel):
    raw_prompt: str = Field(min_length=1, max_length=2000)

    enhance_backend: Literal["ollama", "mock"] = "ollama"
    image_backend: Literal["hf", "mock"] = "mock"

    n_images: int = Field(default=1, ge=1, le=4)

    brand_profile_id: Optional[int] = None


class CreateMLRequestOut(BaseModel):
    request_id: int
    task_id: int
    status: str


class PredictionOut(BaseModel):
    image_path: str
    rank: Optional[int] = None
    clip_score: Optional[float] = None
    aesthetic_score: Optional[float] = None
    final_score: Optional[float] = None

    class Config:
        from_attributes = True


class MLRequestShortOut(BaseModel):
    id: int
    raw_prompt: str
    cleaned_prompt: Optional[str]
    enhanced_prompt: Optional[str]
    status: str
    cost_rub: int

    class Config:
        from_attributes = True


class MLRequestOut(MLRequestShortOut):
    predictions: List[PredictionOut] = Field(default_factory=list)
    best: Optional[PredictionOut] = None


class MLHistoryOut(BaseModel):
    requests: List[MLRequestShortOut]


class MLTaskOut(BaseModel):
    id: int
    request_id: int
    status: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]]
    error: Optional[str]

    class Config:
        from_attributes = True
