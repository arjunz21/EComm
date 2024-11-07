import pytest
from api import app
from fastapi.testclient import TestClient


class Test_Emart:
    client = TestClient(app)

    def test_hello(self):
        response = self.client.get("/test")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
        assert response.json() == {"test": "hello"}
    
    def test_getdata(self):
        response = self.client.get("/getdata")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    def test_transformdata(self):
        response = self.client.get("/transformdata")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    def test_getbestfeatures(self):
        response = self.client.get("/getBestFeatures")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    def test_getbestmodel(self):
        response = self.client.get("/getBestModel")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    def test_train(self):
        response = self.client.get("/train")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    def test_predict(self):
        response = self.client.get("/predict")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
    
    def test_predict(self):
        response = self.client.get("/scores")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
