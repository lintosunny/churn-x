from ml.components.data_ingestion import DataIngestion
from ml.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig

training_pipeline_config = TrainingPipelineConfig()
data_ingestion_config = DataIngestionConfig(training_pipeline_config)

data_ingestion = DataIngestion(data_ingestion_config)


if __name__ == '__main__':
    data_ingestion.initiate_data_ingestion()