import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
# Used to combine preprocessing for numeric and categorical features in pipelines.
from sklearn.impute import SimpleImputer
# Replace missing values with strategies like mean, median, or constant.
from sklearn.pipeline import Pipeline
# used to build machine learning pipelines for chaining preprocessing steps and modeling.
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# The above 2 are used for feature transformation:
# OneHotEncoder: Encodes categorical variables into one-hot numeric arrays.
# StandardScaler: Standardizes numeric features by removing the mean and scaling to unit variance.
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

# Pickle in Python is primarily used in serializing and deserializing a Python object structure. 
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=["writing score","reading score"]
            categorical_columns=[
                "gender", 
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical Columns encoding completed")

            cat_pipeline=Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ]
            )

            logging.info("Categorical Columns encoding completed")

            #Cinnecting the categorical and numerical pipeline
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("categorical_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
            
    def initiateDataTransformation(self, train_path, test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading of train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math score" 
            numerical_columns=["writing score","reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ] 
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
         

  