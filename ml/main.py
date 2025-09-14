from ml.components.data_ingestion import DataIngestion
from ml.components.data_validation import DataValidation
from ml.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from ml.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


if __name__ == '__main__':
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    data_validation_config = DataValidationConfig(training_pipeline_config)
    data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
    data_validation.initiate_data_validation()