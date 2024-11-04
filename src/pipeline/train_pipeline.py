# import os
import sys
from src.exception import CustomException
from src.logger import logging
# import pandas as pd

# from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class TrainPipeline:
    def __init__(self):
        pass

    def initiate_training(self):
        try:
            logging.info("initiated training pipeline")
            data_ingestion_obj = DataIngestion()
            raw_path, train_path, test_path = data_ingestion_obj.initiate_data_ingestion()

            data_transformation_obj = DataTransformation()
            train_array, test_array, preprocessor_path = data_transformation_obj.initiate_data_transformation(train_path=train_path,test_path=test_path)

            model_trainer_object = ModelTrainer()
            r2_square = model_trainer_object.initiate_model_trainer(train_array=train_array,test_array=test_array,preprocessor_path=preprocessor_path)

            print(r2_square)
            

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    train_pipeline_obj = TrainPipeline()
    train_pipeline_obj.initiate_training()