from fastapi import FastAPI, status
from emartapi.models import dbModels, engine
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware import Middleware
# from starlette.middleware.cors import CORSMiddleware
# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

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

# app routes
app.include_router(emart_router)


@app.get("/test/", status_code=status.HTTP_200_OK)
async def test():
    di = DataIngestion()
    tr, val, te = di.start()
    dt = DataTransformation(tr, te, val, "Age", "Sales")
    Xtr, ytr, Xval, yval, Xte, yte, preprocesspath = dt.start()
    m = PriceModel(Xtr, ytr, Xval, yval, preprocesspath)
    bestmodel, mpath = m.getBestModel()
    print("BestModel: ", bestmodel)
    print("Predict: ", PriceModel.predict(preprocesspath, mpath, Xte, yte))
    # print(dt.start())
    return {"test": "test"}
