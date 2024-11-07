import os
import pickle
import configparser
from utils import logging

config = configparser.RawConfigParser()
config.read(os.path.join('resources', 'config.ini'))

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
            with open(filePath, "rb") as fileObj:
                obj = pickle.load(fileObj)
            return obj
        
        except Exception as e:
            logging.info("Error occured in loading object: ", e)
    
    @staticmethod
    def read_config(section, key):
        return config.get(section, key, fallback=None)