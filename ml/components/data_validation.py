import os 
import sys 
import pandas as pd
from distutils import dir_util
from scipy.stats import ks_2samp
from ml.logger import logging
from ml.exception import TelcoChurnException
from ml.utils.main_utils import read_yaml_file, write_yaml_file
from ml.constants.training_pipeline import SCHEMA_FILE_PATH
from ml.entity.config_entity import DataValidationConfig
from ml.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            else:
                return False
        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    def is_numerical_columns_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns
            numerical_column_present = True
            missing_numerical_columns = []
            for num_col in numerical_columns:
                if num_col not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_col)
            logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")
            return numerical_column_present
        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True 
                    status = False 

                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status": is_found
                }}) 

                drift_report_file_path = self.data_validation_config.drift_report_file_path
                dir_path = os.path.dirname(drift_report_file_path)
                os.makedirs(dir_path, exist_ok=True)
                write_yaml_file(file_path=drift_report_file_path, content=report)
                return status
        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    def initiate_data_validation(self) -> DataIngestionArtifact:
        try:
            error_message = ""
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Reading test and train file path 
            train_df = DataValidation.read_data(train_file_path)
            logging.info(f"Read training data from {train_file_path}")
            test_df = DataValidation.read_data(test_file_path)
            logging.info(f"Read test data from {train_file_path}")

            # Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                error_message = f"{error_message} Train dataframe does not contain all the columns. \n"
            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                error_message = f"{error_message} Test dataframe does not contain all the columns. \n"

            # Validate numerical columns
            status = self.is_numerical_columns_exist(dataframe=train_df)
            if not status:
                error_message = f"{error_message} Train dataframe does not contain all the numerical columns. \n"
                status = self.is_numerical_columns_exist(dataframe=test_df)
            if not status:
                error_message = f"{error_message} Test dataframe does not contain all the numerical columns. \n"

            if len(error_message) > 0:
                raise Exception(error_message)
            
            # Check the drift report
            status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)
            logging.info(f"Drift report status: {status}")

            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_ingestion_artifact.train_file_path,
                valid_test_file_path = self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path = None, 
                invalid_test_file_path = None, 
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise TelcoChurnException(e, sys)