import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class DataIngestionConfig:
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.DataPath = r"C:\Users\USER\Documents\projects\Notebooks\data\StudentsPerformance.csv"

    def initiate_data_ingestion(self):
        try:
            logging.info("Entered the data ingestion method or component")
            df = pd.read_csv(self.DataPath)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)

            train_set, test_set = train_test_split(df,test_size=0.2, random_state=14)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)            
            logging.info("Ingestion from csv has been completed")

            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    raw_path, train_path, test_path = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_array, test_array, preprocessor_path = data_transformation_obj.initiate_data_transformation(train_path=train_path,test_path=test_path)

    model_trainer_object = ModelTrainer()
    r2_square = model_trainer_object.initiate_model_trainer(train_array=train_array,test_array=test_array,preprocessor_path=preprocessor_path)

    print(r2_square)
