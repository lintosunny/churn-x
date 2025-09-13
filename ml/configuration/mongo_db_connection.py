import os
import pymongo
import certifi
from ml.constants.database import DATABASE
from dotenv import load_dotenv


load_dotenv()
ca = certifi.where()


class MongoDBClient:
    client = None
    def __init__(self, database=DATABASE) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv("MONGO_DB_URL")
                if "localhost" in mongo_db_url:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                else:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
                self.client = MongoDBClient.client
                self.database = self.client[database]
                self.database_name = database

        except Exception as e:
            raise e