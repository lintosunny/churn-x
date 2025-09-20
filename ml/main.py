# from ml.components.data_ingestion import DataIngestion
# from ml.components.data_validation import DataValidation
# from ml.components.data_transformation import DataTransformation
# from ml.components.model_trainer import ModelTrainer
# from ml.components.model_evaluation import ModelEvaluation
# from ml.components.model_pusher import ModelPusher
# from ml.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
# from ml.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact


# if __name__ == '__main__':
#     training_pipeline_config = TrainingPipelineConfig()
#     data_ingestion_config = DataIngestionConfig(training_pipeline_config)
#     data_ingestion = DataIngestion(data_ingestion_config)
#     data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

#     data_validation_config = DataValidationConfig(training_pipeline_config)
#     data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
#     data_validation_artifact = data_validation.initiate_data_validation()

    
#     data_transformation_config = DataTransformationConfig(training_pipeline_config)
#     data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
#     data_transformation_artifact = data_transformation.initiate_data_transformation()

#     model_trainer_config = ModelTrainerConfig(training_pipeline_config)
#     model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
#     model_trainer_artifact = model_trainer.initiate_model_training()

#     model_evaluation_config = ModelEvaluationConfig(training_pipeline_config)
#     model_evaluation = ModelEvaluation(model_evaluation_config, data_validation_artifact, model_trainer_artifact)
#     model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

#     model_pusher_config = ModelPusherConfig(training_pipeline_config)
#     model_pusher = ModelPusher( model_pusher_config, model_evaluation_artifact)
#     model_pusher_artifact = model_pusher.initiate_model_pusher()


from ml.pipeline.training_pipeline import TrainingPipeline

training_pipeline = TrainingPipeline()
training_pipeline.run_training_pipeline()
