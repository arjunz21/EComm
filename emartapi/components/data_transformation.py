import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import RANSACRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from category_encoders import TargetEncoder


from utils import logging
from utils.helpers import Helpers
from utils.exception import CustomException

@dataclass
class DataTransformationConfig:
    preProcessorPath: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, trPath, tePath, valPath, targetCol):
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
            logging.info(f"We have {len(self.num_features)} Numerical Features: {self.num_features}")
            logging.info(f"We have {len(self.cat_features)} Categorical Features: {self.cat_features}")

            for col in self.tr_df.columns:
                if (self.tr_df[col].nunique() < 30):
                    #print(f"\nWe have {self.tr_df[col].nunique()} categories of {col}:\n", self.tr_df[col].unique())
                    pass
        
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
            numPipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            catPipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("target_encoder", TargetEncoder()), ("scaler", StandardScaler())])

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding & standard scaling completed")

            preprocessor = ColumnTransformer([("numPipeline", numPipeline, self.num_features), ("catPipeline", catPipeline, self.cat_features)])
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

            X_te = self.te_df.drop(columns=self.targetCol)
            y_te = self.te_df[self.targetCol]
            X_te = preprocessingobj.fit_transform(X_te, y_te)
            #y_te = preprocessingobj.transform(y_te)

            X_val = self.val_df.drop(columns=self.targetCol)
            y_val = self.val_df[self.targetCol]
            X_val = preprocessingobj.fit_transform(X_val, y_val)
            #y_val = preprocessingobj.transform(y_val)
            logging.info("Numerical columns standard scaling completed")

            Helpers.save_object(obj=preprocessingobj, filePath=self.dataTransformConfig.preProcessorPath)
            logging.info("Preprocessing object saved as pickle file")

            return X_tr, y_tr, X_val, y_val, X_te, y_te, self.dataTransformConfig.preProcessorPath
        
        except Exception as e:
            logging.error("Error in DataTransformation:" + str(e))
            raise CustomException(e, sys)
