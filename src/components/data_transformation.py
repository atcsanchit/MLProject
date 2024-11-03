import sys
from dataclasses import dataclass
import pandas as pd
import os
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.train_data_path = r"artifacts\train.csv"
        self.test_data_path = r"artifacts\test.csv"
        self.data_transformation_obj = DataTransformationConfig()

    def get_data_transformer_object(self):
        
        try:
            numerical_columns = ["writing score", "reading score"]
            categotical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categotical_columns)
                ]

            )
            logging.info("get_data_transformer_object executed")
            return preprocessor


        except Exception as e:
            logging.info("get_data_transformer_object error")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = self.reading_csv(self.train_data_path)
            test_df = self.reading_csv(self.test_data_path)

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(file_path = self.data_transformation_obj.preprocessor_obj_file_path,
                        obj = preprocessing_obj)
            

            logging.info("initiate_data_transformation executed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_obj.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("initiate_data_transformation error")
            raise CustomException(e,sys)

    def reading_csv(self, filepath):
        try:
            logging.info("performed reading_csv")
            df = pd.read_csv(filepath)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
            logging.info("reading_csv error")

    def ordinal_encoding(self, df):
        try:
            logging.info("performed ordinal_encoding")
            encoder = OrdinalEncoder()
            df["race/ethnicity_encoded"] = encoder.fit_transform(df[["race/ethnicity"]])
            df["parental level of education_encoded"] = encoder.fit_transform(df[["parental level of education"]])
            df["lunch_encoded"] = encoder.fit_transform(df[["lunch"]])
            df["test preparation course_encoded"] = encoder.fit_transform(df[["test preparation course"]])

            return df

        except Exception as e:
            raise CustomException(e,sys)
        logging.info("ordina_encoding error")
    def transformer(self):
        try:
            train_df = self.reading_csv(filepath=self.train_data_path)
            test_df = self.reading_csv(filepath=self.test_data_path)

        except Exception as e:
            raise CustomException(e,sys)
