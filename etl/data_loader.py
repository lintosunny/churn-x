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

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

ca=certifi.where()

class TelcoDataExtract:
    """
    Extracting Telco churn data from CSV files and inserting into MongoDB

    Attributes:
        mongo_client: MongoDB client instance
    """
    def __init__(self, mongo_url: Optional[str] = None):
        """
        Initialize the TelcoDataExtract instance

        Args:
            mongo_url: MongoDB connection URL. If None, uses MONGO_DB_URL constant
        """
        try:
            self.mongo_url = mongo_url or MONGO_DB_URL
            self.mongo_client = None 
            logging.info("TelcoDataExtract initiated successfully")
        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    def _get_mongo_client(self) -> pymongo.MongoClient:
        """Get or create MongoDB client connection."""
        if self.mongo_client is None:
            self.mongo_client = pymongo.MongoClient(self.mongo_url)
        return self.mongo_client
    
    def csv_to_json_convertor(self, file_path: str) -> Dict:
        """
        Convert CSV file to JSON records
            
        Args:
            file_path: Path to the CSV file
                
        Returns:
            List of dictionaries representing JSON records
        """
        try:
            # Validate file path
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            if not path.suffix.lower() == '.csv':
                raise ValueError(f"File must be a CSV file: {file_path}")
            
            logging.info(f"Reading CSV file: {file_path}")

            # Read csv
            data = pd.read_csv(file_path)
            if data.empty:
                raise ValueError(f"CSV file is empty: {file_path}")
            data.reset_index(drop=True, inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            logging.info((f"Successfully converted {len(records)} records from CSV"))
            return records

        except Exception as e:
            logging.error(f"Error converting CSV to JSON: {str(e)}")
            raise TelcoChurnException(e, sys)
        
    def insert_data_mongodb(self, records, database, collection):
        """
        Insert records into MongoDB collection
        
        Args:
            records: List of dictionaries to insert
            database: Database name
            collection: Collection name
            
        Returns:
            Number of records successfully inserted
        """
        try:
            # Validate inputs
            if not records:
                raise ValueError("No records provided for insertion")
            if not database or not collection:
                raise ValueError("Database and collection names must be provided")
            logging.info(f"Inserting {len(records)} records into {database}.{collection}")

            # Get mongodb connection
            mongo_client = self._get_mongo_client()
            db = mongo_client[database]
            coll = db[collection]

            # Insert data
            coll.insert_many(records)
            records_inserted = len(records)
            logging.info(f"Successfully inserted {records_inserted} records")
        
        except Exception as e:
            logging.error(f"Error inserting records to MongoDB: {str(e)}")
            raise TelcoChurnException(e, sys)