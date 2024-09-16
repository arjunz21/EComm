from emartapi.models import base
from emartapi.models import Boolean, Column, Integer, String, TIMESTAMP

class User(base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True)
    dated = Column(TIMESTAMP)

# class Models(base):
#     __tablename__ = "models"
#     name = Column(String(50), unique=True)
#     metric = Column(String(50), unique=True)
#     score = Column(Integer)
