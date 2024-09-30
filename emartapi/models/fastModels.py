from pydantic import BaseModel

class UserModel(BaseModel):
    username: str

class TransformModel(BaseModel):
    trainpath: str
    testpath: str
    targetcolumn: str

class MetricModel(BaseModel):
    name: str
    metric: str
    score: int
