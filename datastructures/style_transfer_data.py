from typing import Optional
from unicodedata import name
from pydantic import BaseModel


class StyleTransferData(BaseModel):
    motion_generation: bool
    file1: str
    file2: str
