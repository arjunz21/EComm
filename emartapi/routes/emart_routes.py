from fastapi import APIRouter, status

emart_router = APIRouter(
    prefix='/api/emart',
    tags=['EMart API']
)

@emart_router.get("/scores/", status_code=status.HTTP_200_OK)
async def scores():
    return {"hello": "scores"}

@emart_router.post("/predict", status_code=status.HTTP_200_OK)
async def predict():
    return {"hello": "predict"}

@emart_router.post("/train", status_code=status.HTTP_200_OK)
async def train():
    return {"hello": "train"}
