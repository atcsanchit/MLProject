import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


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
    data_ingestion_obj.initiate_data_ingestion()