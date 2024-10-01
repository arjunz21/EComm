import os
import pickle

from utils import logging

class Helpers:
    
    @staticmethod
    def save_object(filePath, obj):
        try:
            logging.info("Saving the object")
            dirPath = os.path.dirname(filePath)
            os.makedirs(dirPath, exist_ok=True)
            with open(filePath, "wb") as fileObj:
                pickle.dump(obj, fileObj)
        
        except Exception as e:
            logging.info("Error occured in saving object: ", e)
    
    @staticmethod
    def load_object(filePath):
        try:
            logging.info("Loading the object")
            with open(filePath, "wb") as fileObj:
                obj = pickle.load(fileObj)
            return obj
        
        except Exception as e:
            logging.info("Error occured in loading object: ", e)