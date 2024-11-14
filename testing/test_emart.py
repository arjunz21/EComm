import pytest
from api import app
from fastapi.testclient import TestClient


class Test_Emart:
    client = TestClient(app)

    @pytest.mark.skip
    def test_hello(self):
        response = self.client.get("/api/emart/test")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
        assert response.json() == {"test": "hello api"}
    
    def test_getdata(self):
        response = self.client.get("/api/emart/getdata")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    def test_transformdata(self):
        payload = {
            "trainpath": "artifacts\\train.csv",
            "valpath": "artifacts\\val.csv",
            "testpath": "artifacts\\test.csv",
            "ordcolumn": "Age",
            "targetcolumn": "Sales"}
        response = self.client.post("/api/emart/transformdata", json=payload)
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    def getbestfeatures(self):
        response = self.client.get("/api/emart/getBestFeatures")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    @pytest.mark.skip
    def test_getbestmodel(self):
        payload = {
            "Xtrpath": "artifacts\\Xtr.pkl",
            "ytrpath": "artifacts\\ytr.pkl",
            "Xvalpath": "artifacts\\Xval.pkl",
            "yvalpath": "artifacts\\yval.pkl",
            "prepath": "artifacts\\preprocessor.pkl"}
        response = self.client.post("/api/emart/getBestModel", json=payload)
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    def test_train(self):
        payload = {
            "Xtrpath": "artifacts\\Xtr.pkl",
            "ytrpath": "artifacts\\ytr.pkl",
            "Xvalpath": "artifacts\\Xval.pkl",
            "yvalpath": "artifacts\\yval.pkl",
            "prepath": "artifacts\\preprocessor.pkl"}
        response = self.client.post("/api/emart/train?modelPath=artifacts%5Cmodel.pkl", json=payload)
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    @pytest.mark.xfail
    def test_predict(self):
        payload = {
            "OrderDay": "13",
            "OrderMonth": "8",
            "Quantity": 2,
            "Currency": "USD",
            "Gender": "Male",
            "Name": "Robbie Miller",
            "City": "Houston",
            "State": "Texas",
            "Country": "United States",
            "Continent": "North America",
            "ProductName": "Contoso 8GB Super-Slim MP3/Video Player M800",
            "Brand": " Contoso",
            "Color": "Pink",
            "Subcategory": "MP4&MP3",
            "Category": "Audio",
            "Age": "Adult"}
        response = self.client.post("/api/emart/predict?preprocessPath=artifacts%5Cpreprocessor.pkl&modelPath=artifacts%5Cmodel.pkl", json=payload)
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    @pytest.mark.xfail
    def test_getallscores(self):
        response = self.client.get("/api/emart/getallscores")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 404
    
