from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv(override=True)


def get_mongo_collection():
    client = MongoClient(os.getenv("MONGO_DB_URL"))
    db = client[os.getenv("MONGO_DB")]
    return db[os.getenv("MONGO_COLLECTION")]