from typing import Annotated
from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Boolean, Column, Integer, String, TIMESTAMP, DateTime

# URL_DB = "mysql+pymysql://root@localhost:3306/emart"
URL_DB = "sqlite:///emart.db"
engine = create_engine(URL_DB, connect_args={}, echo=False)
sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
base = declarative_base()

def getDB():
    db = sessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

dbDependency = Annotated[Session, Depends(getDB)]