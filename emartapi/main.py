from fastapi import FastAPI, HTTPException, Depends, status
from typing import Annotated
from models import dbModels, engine
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware import Middleware
# from starlette.middleware.cors import CORSMiddleware
# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app = FastAPI()
dbModels.base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://0.0.0.0', 'http://0.0.0.0:8000',
                   'http://fedora', 'http://fedora:8000',
                   'https://emart.onrender.com', 'https://emart".onrender.com:443',
                   'http://objective-violet-87944.pktriot.net:22010', ],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*']
)

from routes.emart_routes import emart_router

# app routes
app.include_router(emart_router)


@app.get("/helloworld/", status_code=status.HTTP_200_OK)
async def helloworld():
    return {"hello": "Hello World"}
