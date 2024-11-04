import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts","model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")


    def predict(self,features):
        try:
            model = load_object(file_path = self.model_path)
            preprocessor = load_object(file_path = self.preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred    
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 gender,
                 race_ethnicity,
                 parental_level_of_education,
                 lunch,
                 test_preparation_course,
                 reading_score,
                 writing_score
                 ):
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    data=CustomData(
            gender="Male",
            race_ethnicity="group A",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="completed",
            reading_score=80.0,
            writing_score=80.0

        )
    pred_df=data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline=PredictPipeline()
    print("Mid Prediction")
    results=predict_pipeline.predict(pred_df)
    print(results)
    print("after Prediction")
    print(results[0])