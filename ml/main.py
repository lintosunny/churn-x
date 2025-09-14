from ml.components.data_ingestion import DataIngestion
from ml.components.data_validation import DataValidation
from ml.components.data_transformation import DataTransformation
from ml.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from ml.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact


if __name__ == '__main__':
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    data_validation_config = DataValidationConfig(training_pipeline_config)
    data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
    data_validation_artifact = data_validation.initiate_data_validation()

    
    data_transformation_config = DataTransformationConfig(training_pipeline_config)
    data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()