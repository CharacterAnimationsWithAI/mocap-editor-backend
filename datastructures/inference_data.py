from typing import Optional
from unicodedata import name
from pydantic import BaseModel


class InferenceData(BaseModel):
    motion_generation: bool
    file1: str
    file2: Optional[str] = None
