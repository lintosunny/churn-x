import os 
import sys 
import pandas as pd
from pandas import DataFrame
from ml.logger import logging
from ml.exception import TelcoChurnMLException
from ml.data_access.telco_data import TelcoData
from ml.entity.artifact_entity import DataIngestionArtifact
from ml.entity.config_entity import DataIngestionConfig
from ml.utils.main_utils import read_yaml_file
from sklearn.model_selection import train_test_split
from ml.constants.training_pipeline import SCHEMA_FILE_PATH
from ml.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise TelcoChurnMLException(e, sys)
        
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export mongo db collection record as data frame into feature
        """
        try:
            logging.info("Exporting data from MongoDB to feature store")
            telco_data = TelcoData()
            df = telco_data.export_collection_as_df(database_name=DATA_INGESTION_DATABASE_NAME, collection_name=DATA_INGESTION_COLLECTION_NAME)
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            
            # Creating folder
            dir_path = os.path.dirname(feature_store_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(feature_store_path, index=False, header=True)

            logging.info(f"data saved to feature store: {feature_store_path}")
            return df 
        
        except Exception as e:
            raise TelcoChurnMLException(e, sys)
        
    def save_data_split(self, dataframe: DataFrame) -> None:
        """
        Feature store dataset will be split into train and test file
        """
        try:
            train_df, test_df = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train-test split in dataframe")
            
            # Ensure directories exist
            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Started train_df and test_df data exporting...")

            # Save train and test datasets
            train_df.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            logging.info(f"train df saved in {self.data_ingestion_config.train_file_path}")

            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info(f"test df saved in {self.data_ingestion_config.test_file_path}")
        
        except Exception as e:
            raise TelcoChurnMLException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Initiating data ingestion process...")

            # Step 1: Export data
            df = self.export_data_into_feature_store()

            # Step 2: Drop unnecessary columns as per schema
            df = df.drop(self._schema_config["drop_cols"], axis=1)

            # Step 3: Split and save train/test datasets
            self.save_data_split(dataframe=df)

            # Step 4: Create artifact
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

            logging.info("Data ingestion completed successfully")
            return data_ingestion_artifact
        
        except Exception as e:
            raise TelcoChurnMLException(e, sys)