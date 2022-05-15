from typing import Optional
from unicodedata import name
from pydantic import BaseModel


class MotionGenerationData(BaseModel):
    filename: str
    seed_frames: int
