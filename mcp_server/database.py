from sqlalchemy import create_engine, MetaData, Table
from pymongo import MongoClient
from mcp_server.config import settings
from mcp_server.logging import logger

class DatabaseManager:
    def __init__(self):
        self.pg_engine = None
        self.mongo_client = None
        self.metadata = None
        self.tables = {}

    def initialize(self):
        # PostgreSQL
        self.pg_engine = create_engine(settings.database_url, echo=settings.DEBUG, pool_pre_ping=True)
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.pg_engine)
        self.tables = {t: Table(t, self.metadata, autoload_with=self.pg_engine) for t in self.metadata.tables}

        # MongoDB
        self.mongo_client = MongoClient(settings.MONGO_URL)
        self.mongo_client.admin.command('ping')
        logger.info("Databases initialized successfully")

    def get_pg_connection(self):
        return self.pg_engine.connect()

    def get_table(self, table_name):
        return self.tables.get(table_name)

    def get_mongo_collection(self, name=None):
        name = name or settings.MONGO_COLLECTION
        return self.mongo_client[settings.MONGO_DB][name]

db_manager = DatabaseManager()