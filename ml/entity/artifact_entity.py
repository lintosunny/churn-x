from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str 
    test_file_path: str 


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str 
    valid_test_file_path: str 
    invalid_train_file_path: str 
    invalid_test_file_path: str 
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str 
    transformed_train_file_path: str 
    transformed_test_file_path: str


@dataclass 
class ClassificationMetricArtifact:
    f1_score: float 
    precision_score: float 
    recall_score: float 
    recall_score_0: float 
    recall_score_1: float


@dataclass
class ModelTrainerArtifact:
    is_model_accepted: bool 
    improved_accuracy: float 
    best_model_path: str 
    trained_model_path: str 
    trained_model_metric_artifact: str 
    best_model_metric_artifact: str