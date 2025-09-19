import os
import sys 
import json 
import pymongo
import pandas as pd 
import numpy as np 
from typing import Optional
from ml.constants.database import DATABASE
from ml.exception import TelcoChurnMLException
from ml.configuration.mongo_db_connection import MongoDBClient
from ml.logger import logging
from dotenv import load_dotenv
load_dotenv()


MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class TelcoData:
    """
    Export entire MongoDB record as pandas dataframe
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
            raise TelcoChurnMLException(e, sys)
        
    def _get_mongo_client(self) -> pymongo.MongoClient:
        """Get or create MongoDB client connection."""
        if self.mongo_client is None:
            self.mongo_client = pymongo.MongoClient(self.mongo_url)
        return self.mongo_client
        
    def export_collection_as_df(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Export all documents from a MongoDB collection into a Pandas DataFrame

        Args:
            collection_name (str): The name of the MongoDB collection to export
            database_name (Optional[str], default=None): The name of the MongoDB database

        Returns:
            pd.DataFrame: A DataFrame containing all records from the collection
        """
        try:
            if not database_name or not collection_name:
                raise ValueError("Database and Collection name must be provided")

            # Get mongodb connection
            mongo_client = self._get_mongo_client()
            database = mongo_client[database_name]
            collection = database[collection_name]

            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Importing {len(df)} records from MongoDB")

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            
            df.replace({"na": np.nan}, inplace=True)

            return df
        
        except Exception as e:
            raise TelcoChurnMLException(e, sys)