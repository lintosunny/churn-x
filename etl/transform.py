import os 
import sys 
import json
import pymongo
import certifi
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from ml.exception import TelcoChurnException
from ml.logger import logging
from etl.constants import RAW_FILE_PATH, TRANSFORMED_FILE_PATH

class TelcoDataTransform:
    def __init__(self, extract_from: str = RAW_FILE_PATH, load_to: str = TRANSFORMED_FILE_PATH):
        try:
            self.extract_from = extract_from
            self.load_to = load_to
            logging.info(f"Initialized TelcoDataTransform with extract_from={extract_from}, load_to={load_to}")
        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from {file_path}")
            df = pd.read_csv(file_path)
            logging.info(f"Data read successfully with shape {df.shape}")
            return df 
        except Exception as e:
            logging.error(f"Failed to read data from {file_path}")
            raise TelcoChurnException(e, sys)
        
    def create_columns(self) -> pd.DataFrame:
        try:
            logging.info("Starting feature creation process")
            df = TelcoDataTransform.read_data(self.extract_from)

            # col 1: Additional services
            additional_service_columns = ["Online Security", "Online Backup", "Device Protection", "Tech Support"]
            logging.info(f"Creating 'additional_services' column using {additional_service_columns}")
            df["additional_services"] = df[additional_service_columns].apply(
                lambda x: (~x.isin(['No', 'No internet service', 'No phone service'])).sum(), axis=1
            ).astype(int)
            
            # col 2: 3rd party services
            third_party_service_columns = ["Streaming TV", "Streaming Movies"]
            logging.info(f"Creating '3rd_party_services' column using {third_party_service_columns}")
            df["3rd_party_services"] = df[third_party_service_columns].apply(
                lambda x: (~x.isin(['No', 'No internet service', 'No phone service'])).sum(), axis=1
            ).astype(int)

            # col 3: is_payment_automatic
            logging.info("Creating 'is_payment_automatic' column")
            df['is_payment_automatic'] = df['Payment Method'].str.contains('Automatic', case=False, na=False).astype(int)

            # col 4: is_payment_recurring
            logging.info("Creating 'is_payment_recurring' column")
            df['is_payment_recurring'] = df['Contract'].map({"Month-to-month": 1, "One year": 0, "Two year": 0}).astype(int)
            
            logging.info("Feature creation completed successfully")
            return df
        
        except Exception as e:
            logging.error("Failed during feature creation")
            raise TelcoChurnException(e, sys)
        
    def save_transformed_dataframe(self, dataframe: pd.DataFrame) -> None:
        try:
            logging.info(f"Saving transformed dataframe to {self.load_to}")
            dataframe.to_csv(self.load_to, index=False, header=True)
            logging.info("Data saved successfully")
        except Exception as e:
            logging.error(f"Failed to save transformed dataframe to {self.load_to}")
            raise TelcoChurnException(e, sys)
        
    def initiate_data_transformation(self):
        try:
            logging.info("Starting data transformation process")
            df = self.create_columns()
            self.save_transformed_dataframe(df)
            logging.info("Data transformation process completed successfully")
        except Exception as e:
            logging.error("Data transformation process failed")
            raise TelcoChurnException(e, sys)