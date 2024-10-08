from pydantic import BaseModel

class UserModel(BaseModel):
    username: str

class TransformModel(BaseModel):
    trainpath: str = "artifacts\\train.csv"
    valpath: str = "artifacts\\val.csv"
    testpath: str = "artifacts\\test.csv"
    ordcolumn: str = "Age"
    targetcolumn: str = "Sales"

    class Config:
        orm_mode: True

class ModelTrainerModel(BaseModel):
    Xtrpath: str = "artifacts\\Xtr.pkl"
    ytrpath: str = "artifacts\\ytr.pkl"
    Xvalpath: str = "artifacts\\Xval.pkl"
    yvalpath: str = "artifacts\\yval.pkl"
    prepath: str = "artifacts\\preprocessor.pkl"

    class Config:
        orm_mode: True

class MetricModel(BaseModel):
    name: str
    mse: int
    mae: int
    mape: int
    r2: int
    adjr2: int

    class Config:
        orm_mode: True

class PredictModel(BaseModel):
    OrderDay: str = "13"
    OrderMonth: str = "8"
    Quantity: int = 2
    Currency: str = "USD"
    Gender: str = "Male"
    Name: str = "Robbie Miller"
    City: str = "Houston"
    State: str = "Texas"
    Country: str = "United States"
    Continent: str = "North America"
    ProductName: str = "Contoso 8GB Super-Slim MP3/Video Player M800"
    Brand: str = " Contoso"
    Color: str = "Pink"
    Subcategory: str = "MP4&MP3"
    Category: str = "Audio"
    Age: str = "Adult"
    #profit: int = 0

    class Config:
        orm_mode: True
