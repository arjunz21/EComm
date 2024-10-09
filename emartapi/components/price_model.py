import os
import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_regression, RFE, RFECV
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
# from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error, f1_score, median_absolute_error

from utils import logging
from utils.helpers import Helpers


@dataclass
class PriceModelConfig:
    trainedModelPath: str = os.path.join('artifacts', 'model.pkl')
    customModelPath: str = os.path.join('artifacts', 'custom.pkl')


class PriceModel:
    def __init__(self, XTrainPath, yTrainPath, XValPath, yValPath, preprocessorpath):
        self.modelTrainerConfig = PriceModelConfig()
        self.XTrain = Helpers.load_object(XTrainPath)
        self.yTrain = Helpers.load_object(yTrainPath)
        self.XVal = Helpers.load_object(XValPath)
        self.yVal = Helpers.load_object(yValPath)
        self.pre = Helpers.load_object(preprocessorpath)    
    
    def getBestFeatures(self, Xtr, ytr):
        # K-Best for numerical features
        kbest = SelectKBest(score_func=f_regression, k=8)
        fit = kbest.fit(Xtr, ytr)

        # K-Best for categorical features
        chi = SelectKBest(score_func=mutual_info_classif, k=8)
        fit = chi.fit(Xtr, ytr)
        print(set(Xtr.columns[fit.get_support(indices=True)].toList(), Xtr.columns[fit.get_support(indices=True)].toList()))
        
        # RFE
        rfecv = RFECV(estimator=f_regression, step=1, cv=3)
        rfecv.fit(Xtr, ytr)
        print(rfecv.grid_scores)

    def getBestModel(self):
        self.models={
                "Decision Tree": {
                    'cls': DecisionTreeRegressor(),
                    'params': {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'splitter':['best','random'],
                        # 'max_features':['sqrt','log2'],
                    }
                },
                "Random Forest":{
                    'cls': RandomForestRegressor(),
                    'params': {
                        # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'max_features':['sqrt','log2',None],
                        'n_estimators': [8,16,32,64,128,256]
                    }
                },
                "Gradient Boosting":{
                    'cls': GradientBoostingRegressor(),
                    'params': {
                        # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                        'learning_rate':[.1,.01,.05,.001],
                        # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        # 'criterion':['squared_error', 'friedman_mse'],
                        # 'max_features':['auto','sqrt','log2'],
                        'n_estimators': [8,16,32,64,128,256]
                    }
                },
                "Linear Regression":{
                    'cls': LinearRegression(),
                    'params': {}
                },
                "K-Neighbour Regressor":{
                    'cls': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors':[5,7,9,11],
                        # 'weights':['uniform','distance'],
                        # 'algorithm':['ball_tree','kd_tree','brute']
                    }
                },
                "XGBRegressor":{
                    'cls': XGBRegressor(),
                    'params': {
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators': [8,16,32,64,128,256]
                    }
                },
                "CatBoosting Regressor":{
                    'cls': CatBoostRegressor(),
                      'params': {
                        'depth': [6,8,10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                      }
                },
                "AdaBoost Regressor":{
                    'cls': AdaBoostRegressor(),
                    'params': {
                        'learning_rate':[.1,.01, 0.5, .0001],
                        # 'loss':['linear','square','exponential'],
                        'n_estimators': [8,16,32,64,128,256]
                    }
                }
            }

        for name, model in self.models.items():
            print("Model: ", name)
            self.models[name]['name'] = name
            cls = model['cls']
            params = model['params']

            # Finding the best params for the models with KFold & GridSearchCV
            kf = KFold(3, shuffle=True, random_state=1)
            gs = GridSearchCV(estimator=cls, param_grid=params, cv=kf)
            gs.fit(self.XTrain, self.yTrain)
            
            # Fitting with XTrain & YTrain Data
            cls.set_params(**gs.best_params_)
            cls.fit(self.XTrain, self.yTrain)

            ypreds = cls.predict(self.XTrain)
            ytest_preds = cls.predict(self.XVal)
            self.models[name]['best_params'] = gs.best_params_
            self.models[name]['tr_mse'], self.models[name]['tr_mae'], self.models[name]['tr_mape'], self.models[name]['tr_r2'], self.models[name]['tr_adjr2'] = self.scores(ypreds, self.yTrain)
            self.models[name]['tval_mse'], self.models[name]['tval_mae'], self.models[name]['tval_mape'], self.models[name]['tval_r2'], self.models[name]['tval_adjr2'] = self.scores(ytest_preds, self.yVal)
                    
        ## To get best model score from dict
        best_model = pd.DataFrame(self.models).T.sort_values(by=['tr_r2', 'tval_r2', 'tr_mae', 'tval_mae'], ascending=[False, False, False, False]).iloc[0]
        Helpers.save_object(filePath=self.modelTrainerConfig.trainedModelPath, obj=best_model['cls'])
        best_model['cls'] = ""
        print("BestModel: ", best_model)

        if best_model['tr_r2'] < 0.7 and best_model['tval_r2'] < 0.7:
             logging.error("No Best model Found above R2 score of 0.7")
        # logging.info("BestModel: ", best_model.to_dict())
        
        logging.info('Best model found on both training and testing dataset')
        return best_model.to_dict(), self.modelTrainerConfig.trainedModelPath

    def train(self, modelPath):
        # Loading model & Data
        model = Helpers.load_object(modelPath)
        
        # Training Model
        model.fit(self.XTrain, self.yTrain)
        Helpers.save_object(filePath=self.modelTrainerConfig.customModelPath, obj=model)
        ypreds = model.predict(self.XTrain)
        yval_preds = model.predict(self.XVal)

        # Checking for Training & Test scores
        tr_mse, tr_mae, tr_mape, tr_r2, tr_adjr2 = self.scores(ypreds, self.yVal)
        tval_mse, tval_mae, tval_mape, tval_r2, tval_adjr2 = self.scores(yval_preds, self.yVal)
        return { 'scores': {'tr_mse': tr_mse, 'tr_mae': tr_mae, 'tr_mape': tr_mape, 'tr_r2': tr_r2, 'tr_adjr2': tr_adjr2,
                 'tval_mse': tval_mse, 'tval_mae': tval_mae, 'tval_mape': tval_mape, 'tval_r2': tval_r2, 'tval_adjr2': tval_adjr2},
                 'ypreds': ypreds, 'ytest_preds': yval_preds, 'modelPath': self.modelTrainerConfig.customModelPath }
         
    @staticmethod
    def predict(preProcessPath, modelPath, Xte, yte=0):
        model = Helpers.load_object(modelPath)
        preprocessor = Helpers.load_object(preProcessPath)
        Xte = preprocessor.transform(pd.DataFrame([Xte]))
        preds = model.predict(Xte)
        return {'scores':'self.scores(preds, yte)', 'preds':str(preds)}
    
    def scores(self, ypreds, ytest):
        mse = mean_squared_error(ytest, ypreds)
        mae = median_absolute_error(ytest, ypreds)
        mape=np.mean(np.abs((ytest - ypreds)/ytest))*100
        r2 = r2_score(ytest, ypreds)
        adj_r2 = 1 - ((1 - r2) * (self.XTrain.shape[0]-1) / (self.XTrain.shape[0] - self.XTrain.shape[1] - 1))
        return round(mse, 3), round(mae, 3), round(mape, 3), round(r2, 3), round(adj_r2, 3)
        
    def __str__(self):
        pass