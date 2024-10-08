import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import RANSACRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from category_encoders import TargetEncoder

from utils import logging
from utils.helpers import Helpers
from utils.exception import CustomException

#set_config(display='diagram')

@dataclass
class DataTransformationConfig:
    preProcessorPath: str = os.path.join('artifacts', 'preprocessor.pkl')
    XtrPath: str = os.path.join('artifacts', 'Xtr.pkl')
    ytrPath: str = os.path.join('artifacts', 'ytr.pkl')
    XvalPath: str = os.path.join('artifacts', 'Xval.pkl')
    yvalPath: str = os.path.join('artifacts', 'yval.pkl')
    XtePath: str = os.path.join('artifacts', 'Xte.pkl')
    ytePath: str = os.path.join('artifacts', 'yte.pkl')


class DataTransformation:
    def __init__(self, trPath, tePath, valPath, ordCol, targetCol):
        self.dataTransformConfig = DataTransformationConfig()
        self.targetCol = targetCol
        try:
            logging.info("Reading Train & Test data completed")
            self.tr_df = pd.read_csv(trPath)
            self.te_df = pd.read_csv(tePath)
            self.val_df = pd.read_csv(valPath)

            logging.info("Dropping null values & duplicates in Dataframe")
            self.tr_df.dropna(how="all", inplace=True)
            self.tr_df.drop_duplicates(inplace=True)
            self.te_df.dropna(how="all", inplace=True)
            self.te_df.drop_duplicates(inplace=True)
            self.val_df.dropna(how="all", inplace=True)
            self.val_df.drop_duplicates(inplace=True)

            self.num_features = list(self.tr_df.select_dtypes(include=["float", "int"]).columns)
            self.num_features.remove(self.targetCol)
            self.cat_features = list(self.tr_df.select_dtypes(include=["object"]).columns)            
            self.ord_features = list(ordCol.split())
            self.cat_features = list(set(self.cat_features) - set(self.ord_features))
            self.ohe_features = []

            for col in self.cat_features:
                if (self.tr_df[col].nunique() <= 5):
                    self.ohe_features.append(col)
                    self.cat_features.remove(col)
                    print(f"\nWe have {self.tr_df[col].nunique()} categories of {col}:\n", self.tr_df[col].unique())
                if (self.tr_df[col].nunique() > 5) and (self.tr_df[col].nunique() < 30):
                    print(f"\nWe have {self.tr_df[col].nunique()} categories of {col}:\n", self.tr_df[col].unique())
            
            # self.num_features = [self.tr_df.columns.get_loc(col) for col in self.num_features]
            logging.info(f"We have {len(self.num_features)} Numerical Features: {self.num_features}")
            logging.info(f"We have {len(self.ohe_features)} OHE Features: {self.ohe_features}")
            logging.info(f"We have {len(self.ord_features)} ORD Features: {self.ord_features}")
            logging.info(f"We have {len(self.cat_features)} Categorical Features: {self.cat_features}")
        except Exception as e:
            logging.error("Error in Converting to DataFrame:" + str(e))
            raise CustomException(e, sys)
    
    
    def handleOutliers(self):
        # Elliptic Envelop
        ee = EllipticEnvelope(contamination=0.01)
        self.tr_df['elliptic'] = ee.fit_predict(self.tr_df)
        
        # IsolationForest
        iso = IsolationForest(contamination=0.01)
        self.tr_df['ISOForest'] = iso.fit_predict(self.tr_df)

        # One-Class SVM
        osvm=OneClassSVM()
        self.tr_df['osvm'] = osvm.fit_predict(self.tr_df)

        # Local Outlier Detector
        lof=LocalOutlierFactor(contamination=0.01)
        self.tr_df['lof'] = lof.fit_predict(self.tr_df)


    def getDataTransformerObj(self):
        try:
            numPipe = Pipeline(steps=[("numimputer", SimpleImputer(strategy="median"))])

            ohePipe = Pipeline(steps=[("oheimputer", SimpleImputer(strategy="most_frequent")),
                                      ("ohe_encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
            
            ordPipe = Pipeline(steps=[("ordimputer", SimpleImputer(strategy="most_frequent")),
                                      ("ord_encoder", OrdinalEncoder())])
            
            catPipe = Pipeline(steps=[("catimputer", SimpleImputer(strategy="most_frequent")),
                                      ("target_encoder", TargetEncoder())])
            
            colTransformer = ColumnTransformer(transformers=[('num_pipeline', numPipe, self.num_features),
                                                             ('ohe_pipeline', ohePipe, self.ohe_features),
                                                             ('ord_pipeline', ordPipe, self.ord_features),
                                                             ('cat_pipeline', catPipe, self.cat_features)], remainder='passthrough')


            preprocessor = Pipeline(steps=[('impute_encode', colTransformer), ('scale', StandardScaler())])
            print(preprocessor.named_steps)
            
            logging.info("Numerical columns standard scaling pipeline created")
            logging.info("Categorical columns encoding & standard scaling pipeline created")
            return preprocessor
        
        except Exception as e:
            logging.error("Error in Creation of GetDataTransformation Object:" + str(e))
            raise CustomException(e, sys)
        

    def start(self):
        try:
            preprocessingobj = self.getDataTransformerObj()
            X_tr = self.tr_df.drop(columns=[self.targetCol], axis=1)
            y_tr = self.tr_df[self.targetCol]
            X_tr = preprocessingobj.fit_transform(X_tr, y_tr)
            #y_tr = preprocessingobj.transform(y_tr)
            Helpers.save_object(obj=X_tr, filePath=self.dataTransformConfig.XtrPath)
            Helpers.save_object(obj=y_tr, filePath=self.dataTransformConfig.ytrPath)

            X_te = self.te_df.drop(columns=self.targetCol)
            y_te = self.te_df[self.targetCol]
            X_te = preprocessingobj.transform(X_te)
            #y_te = preprocessingobj.transform(y_te)
            Helpers.save_object(obj=X_te, filePath=self.dataTransformConfig.XvalPath)
            Helpers.save_object(obj=y_te, filePath=self.dataTransformConfig.yvalPath)

            X_val = self.val_df.drop(columns=self.targetCol)
            print("Xval ", X_val.columns)
            y_val = self.val_df[self.targetCol]
            X_val = preprocessingobj.transform(X_val)
            #y_val = preprocessingobj.transform(y_val)
            Helpers.save_object(obj=X_val, filePath=self.dataTransformConfig.XtePath)
            Helpers.save_object(obj=y_val, filePath=self.dataTransformConfig.ytePath)
            logging.info("Numerical & Categorical columns standard scaling completed")

            Helpers.save_object(obj=preprocessingobj, filePath=self.dataTransformConfig.preProcessorPath)
            logging.info("Preprocessing object saved as pickle file")
            return self.dataTransformConfig.XtrPath, self.dataTransformConfig.ytrPath, self.dataTransformConfig.XvalPath, self.dataTransformConfig.yvalPath, self.dataTransformConfig.XtePath, self.dataTransformConfig.ytePath, self.dataTransformConfig.preProcessorPath
        
        except Exception as e:
            logging.error("Error in DataTransformation:" + str(e))
            raise CustomException(e, sys)


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.label_encoders = {}

    def fit(self, X, y=None):
        print(X)
        if self.cols is None:
            self.cols = X.columns

        for col in self.cols:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col, le in self.label_encoders.items():
            X_encoded[col] = le.transform(X_encoded[col])
        return X_encoded