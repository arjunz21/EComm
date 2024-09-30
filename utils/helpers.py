import os
import pickle

from utils import logging

class Helpers:
    
    @staticmethod
    def save_object(filePath, obj):
        try:
            dirPath = os.path.dirname(filePath)
            os.makedirs(dirPath, exist_ok=True)
            with open(filePath, "wb") as fileObj:
                pickle.dump(obj, fileObj)
        
        except Exception as e:
            logging.info("Error occured in saving file: ", e)