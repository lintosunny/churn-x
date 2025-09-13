import os 


SAVED_MODEL_DIR = os.path.join("ml/saved_models")
"""
Common constant variable for training pipeline
"""
ARTIFACT_DIR: str = "artifact"
PIPELINE_NAME: str = "telcoChurn"
FILE_NAME: str = "telco.csv"
TARGET_COLUMN: str = "CHURN VALUE"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH: str = os.path.join("ml", "configuration", "schema.yaml")

"""
Data ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "telco"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_DATABASE_NAME: str = "CHURNX"
DATA_INGESTION_COLLECTION_NAME: str = "TelcoChurnData"