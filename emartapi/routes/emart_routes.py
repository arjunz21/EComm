from fastapi import APIRouter, status

import pandas as pd
from emartapi import models
from emartapi.models.fastModels import TransformModel, PredictModel, MetricModel, ModelTrainerModel
from emartapi.models import Session, Depends

from emartapi.components import DataIngestion
from emartapi.components.data_transformation import DataTransformation
from emartapi.components.price_model import PriceModel

emart_router = APIRouter(
    prefix='/api/emart',
    tags=['EMart API']
)

@emart_router.get("/getdata", status_code=status.HTTP_200_OK)
async def getData():
    di = DataIngestion()
    return di.start()

@emart_router.post("/transformdata", status_code=status.HTTP_200_OK)
async def transformdata(data: TransformModel):
    dt = DataTransformation(trPath=data.trainpath, valPath=data.valpath, ordCol=data.ordcolumn, targetCol=data.targetcolumn)
    return dt.start()

@emart_router.post("/getBestFeatures", status_code=status.HTTP_200_OK)
async def getBestFeatures(db: Session = Depends(models.getDB)):
    return {"hello": "BestFeatures"}

@emart_router.post("/getBestModel", status_code=status.HTTP_200_OK)
async def getBestModel(data: ModelTrainerModel, db: Session = Depends(models.getDB)):
    m = PriceModel(data.Xtrpath, data.ytrpath, data.Xvalpath, data.yvalpath, data.prepath)
    return m.getBestModel()

@emart_router.post("/train", status_code=status.HTTP_200_OK)
async def train(data: ModelTrainerModel, modelPath: str="artifacts\\model.pkl", db: Session = Depends(models.getDB)):
    m = PriceModel(data.Xtrpath, data.ytrpath, data.Xvalpath, data.yvalpath, data.prepath)
    return m.train(modelPath)

@emart_router.get("/scores", response_model=MetricModel, status_code=status.HTTP_200_OK)
async def scores(db: Session = Depends(models.getDB)):
    return {"hello": "scores"}

@emart_router.get("/getallscores", status_code=status.HTTP_200_OK)
async def getallscores(db: Session = Depends(models.getDB)):
    return PriceModel.getAllScores()

@emart_router.post("/predict", status_code=status.HTTP_200_OK)
async def predict(inp: PredictModel, preprocessPath: str="artifacts\\preprocessor.pkl",
                  modelPath: str="artifacts\\model.pkl"):
    xte = pd.DataFrame([dict(inp)])
    return PriceModel.predict(preprocessPath, modelPath, xte, ["0"])
