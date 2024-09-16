from pydantic import BaseModel

class UserBase(BaseModel):
    username: str

class ModelsBase(BaseModel):
    name: str
    metric: str
    score: int