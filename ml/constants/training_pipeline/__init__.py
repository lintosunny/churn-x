import os 


SAVED_MODEL_DIR = os.path.join("ml/saved_models")
"""
Common constant variable for training pipeline
"""
ARTIFACT_DIR: str = "artifact"
PIPELINE_NAME: str = "telcoChurn"
FILE_NAME: str = "telco.csv"
TARGET_COLUMN: str = "Churn Value"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.joblib"
MODEL_FILE_NAME: str = "model.joblib"
SCHEMA_FILE_PATH: str = os.path.join("ml", "configuration", "schema.yaml")
HYPER_PARAMETER_TUNING_OPTUNA_FILE_NAME: str = "optuna.json"

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

"""
Data validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
Model trainer related constant start with MODEL_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVERFITTING_UNDER_FITTING_THRESHOLD: float = 0.35