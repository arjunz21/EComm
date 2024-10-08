from datetime import datetime
from emartapi.models import base
from emartapi.models import Boolean, Column, Integer, String, TIMESTAMP, DateTime

class User(base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True)
    dated = Column(TIMESTAMP)

class ModelStats(base):
    __tablename__ = "modelstats"
    mid = Column(Integer, primary_key=True)
    dated = Column(DateTime, nullable=False, default=datetime.utcnow)
    name = Column(String(50), unique=True)
    params = Column(String(50), unique=True)
    trmse = Column(Integer)
    trmae = Column(Integer)
    trmape = Column(Integer)
    trr2 = Column(Integer)
    tradjr2 = Column(Integer)
    tvalmse = Column(Integer)
    tvalmae = Column(Integer)
    tvalmape = Column(Integer)
    tvalr2 = Column(Integer)
    tvaladjr2 = Column(Integer)

    def __repr__(self):
        return f"Model: {self.name}, TR_R2: {self.trr2}, Tval_R2: {self.tvalr2}"
