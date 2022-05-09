from typing import Optional
from unicodedata import name
from pydantic import BaseModel


class MotiongenerationData(BaseModel):
    motion_generation: bool
    file: str
