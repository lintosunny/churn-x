import sys 
import numpy as np
import pandas as pd 
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from category_encoders import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ml.logger import logging
from ml.exception import TelcoChurnException
from ml.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)
from ml.entity.config_entity import DataTransformationConfig
from ml.constants.training_pipeline import SCHEMA_FILE_PATH
from ml.constants.training_pipeline import TARGET_COLUMN
from ml.utils.main_utils import save_numpy_array_data, save_object, read_yaml_file


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise TelcoChurnException(e, sys)
          
    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Fetching column configuration from schema.")
            scaler_cols = self._schema_config["cols_to_scale"]
            target_encode_cols = self._schema_config["cols_to_target_encode"]
            binary_cols = self._schema_config["cols_to_label_encode"]
            
            logging.info(f"Numeric columns to scale: {scaler_cols}")
            logging.info(f"Target encode columns: {target_encode_cols}")
            logging.info(f"Binary (Yes/No) columns: {binary_cols}")

            # Components
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            target_encoder = TargetEncoder()

            # Pipelines
            logging.info("Building numeric pipeline...")
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", simple_imputer),
                    ("scaler", robust_scaler)
                ]
            )

            logging.info("Building binary pipeline...")
            binary_pipeline = Pipeline(
                steps=[
                    ("encoder", OrdinalEncoder(mapping=[
                        {"col": col, "mapping": {"No": 0, "Yes": 1}}
                        for col in binary_cols
                    ]))
                ]
            )

            # Column transformer
            logging.info("Combining pipelines into ColumnTransformer...")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, scaler_cols),
                    ("binary", binary_pipeline, binary_cols),
                    ("target", target_encoder, target_encode_cols)
                ],
                remainder="passthrough"
            )

            logging.info("Preprocessor successfully built.") 
            return preprocessor
        
        except Exception as e:
            logging.error("Error occurred while building preprocessor.", exc_info=True)
            raise TelcoChurnException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation process.")

            # 1. Load train and test datasets
            logging.info("Reading train and test datasets...")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # 2. Separate features and target
            logging.info("Separating features and target column...")
            X_train, y_train = train_df.drop(columns=[TARGET_COLUMN], axis=1), train_df[TARGET_COLUMN]
            X_test, y_test = test_df.drop(columns=[TARGET_COLUMN], axis=1), test_df[TARGET_COLUMN]

            # 3. Get preprocessing pipeline
            logging.info("Building preprocessing pipeline...")
            preprocessor = self.get_data_transformer_object()

            # Fit preprocessor only on training data
            logging.info("Fitting preprocessor on training data...")
            preprocessor_object = preprocessor.fit(X_train, y_train)

            logging.info("Transforming train and test datasets...")
            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info(f"After preprocessing -> Train shape: {X_train_transformed.shape}, Test shape: {X_test_transformed.shape}")

            # 4. Apply SMOTETomek only on training set
            logging.info("Applying SMOTETomek to balance classes in training set...")
            smt = SMOTETomek(sampling_strategy="auto")
            X_train_resampled, y_train_resampled = smt.fit_resample(X_train_transformed, y_train)

            logging.info(f"Resampled training set -> X: {X_train_resampled.shape}, y: {y_train_resampled.shape}")

            # 5. Concatenate features + target
            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save transformed datasets
            logging.info("Saving transformed datasets and preprocessor object...")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            # 6. Create artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        
        except Exception as e:
            logging.error("Error occurred during data transformation.", exc_info=True)
            raise TelcoChurnException(e, sys)