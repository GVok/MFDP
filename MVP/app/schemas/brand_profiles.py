from pydantic import BaseModel, Field
from typing import List


class BrandProfileCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    style_short: str = Field(default="", max_length=2000)


class BrandProfileOut(BaseModel):
    id: int
    name: str
    style_short: str

    class Config:
        from_attributes = True


class BrandProfilesOut(BaseModel):
    profiles: List[BrandProfileOut]
