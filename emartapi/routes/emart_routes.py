from fastapi import APIRouter, status

from emartapi import models
from emartapi.models import fastModels
from emartapi.models import Session, Depends

from emartapi.components.data_ingestion import DataIngestion
from emartapi.components.data_transformation import DataTransformation

emart_router = APIRouter(
    prefix='/api/emart',
    tags=['EMart API']
)

@emart_router.post("/getdata", status_code=status.HTTP_200_OK)
async def getData():
    di = DataIngestion()
    return {"traintest": di.start()}

@emart_router.post("/transformdata", status_code=status.HTTP_200_OK)
async def transformdata(data: fastModels.TransformModel):
    dt = DataTransformation(trPath=data.trainpath, tePath=data.testpath, targetCol=data.targetcolumn)
    X, y, preprocesspath = dt.start()
    return {"preprocesspath": preprocesspath}

@emart_router.post("/train", status_code=status.HTTP_200_OK)
async def train():
    return {"hello": "train"}

@emart_router.post("/predict", status_code=status.HTTP_200_OK)
async def predict():
    return {"hello": "predict"}

@emart_router.get("/scores/", response_model=fastModels.MetricModel, status_code=status.HTTP_200_OK)
async def scores(db: Session = Depends(models.getDB)):
    return {"hello": "scores"}
