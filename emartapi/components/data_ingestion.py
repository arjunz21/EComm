import os
# import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from utils import logging


@dataclass
class DataIngestionConfig:
    curPath: str = os.getcwd()
    rawDataPath: str = os.path.join('artifacts', 'raw.csv')
    trDataPath: str = os.path.join('artifacts', 'train.csv')
    teDataPath: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    def start(self):
        logging.info("Data Ingestion started.")
        try:
            df = pd.read_csv(os.path.join(self.ingestionConfig.curPath, 'Notebooks', 'data', 'EMart.csv'))
            logging.info("Reading the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestionConfig.rawDataPath), exist_ok=True)

            logging.info("Applying train test split")
            tr_set, te_set = train_test_split(df, test_size=0.2, random_state=42)
            df.to_csv(self.ingestionConfig.rawDataPath, index=False, header=True)
            tr_set.to_csv(self.ingestionConfig.trDataPath, index=False, header=True)
            te_set.to_csv(self.ingestionConfig.teDataPath, index=False, header=True)

            logging.info("Data Ingestion completed")
        except Exception as e:
            logging.error("Error in DataIngestion:" + str(e))

        return self.ingestionConfig.trDataPath, self.ingestionConfig.teDataPath