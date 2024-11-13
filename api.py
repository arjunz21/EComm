import pandas as pd
from fastapi import FastAPI, status
from emartapi.models import dbModels, engine
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware import Middleware
# from starlette.middleware.cors import CORSMiddleware
# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

from utils.helpers import Helpers

app = FastAPI()
dbModels.base.metadata.create_all(bind=engine)

app.add_middleware(CORSMiddleware,
    allow_origins=['http://0.0.0.0', 'http://0.0.0.0:8000',
                   'http://fedora', 'http://fedora:8000',
                   'https://emart.onrender.com', 'https://emart".onrender.com:443',
                   'http://objective-violet-87944.pktriot.net:22010', ],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

from emartapi.components import DataIngestion
from emartapi.components.data_transformation import DataTransformation
from emartapi.components.price_model import PriceModel
from emartapi.routes import emart_router
from os import environ as env

# app routes
app.include_router(emart_router)

@app.get("/", status_code=status.HTTP_200_OK)
async def index():
    return {"result": f"hello var env['MY_VAR']"}

@app.get("/test/", status_code=status.HTTP_200_OK)
async def hello():
    target = "Sales"
    di = DataIngestion()
    tr, val, te = di.start()
    dt = DataTransformation(tr, val, "Age", target)
    Xtr, ytr, Xval, yval, preprocesspath, _ = dt.start()
    m = PriceModel(Xtr, ytr, Xval, yval, preprocesspath)
    bestmodel, mpath = m.getBestModel()
    print("BestModel: ", bestmodel)
    te = pd.read_csv(te)
    print("Predict: ", PriceModel.predict(preprocesspath, mpath, te.drop(columns=[target]), te[target]))
    return {"test": "hello api"}
