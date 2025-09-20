from ml.exception import TelcoChurnMLException
from ml.logger import logging
from ml.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from ml.entity.config_entity import ModelPusherConfig, TrainingPipelineConfig
from ml.cloud.s3_syncer import S3Sync
import os 
import sys 
import shutil
from ml.constants.training_pipeline import MODEL_PUSHER_S3_BUCKET_NAME
from dotenv import load_dotenv

load_dotenv(override=True)


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
            self.training_pipeline_config = TrainingPipelineConfig()
            self.s3_syncer = s3_syncer = S3Sync()

        except Exception as e:
            raise TelcoChurnMLException(e, sys)
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            trained_model_path = self.model_evaluation_artifact.trained_model_path
            best_model_path = self.model_evaluation_artifact.best_model_path 

            # Creating model pusher dir to save model
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)

            # saved model dir
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            shutil.copy(src=best_model_path, dst=saved_model_path)

            # Prepare artifact
            model_pusher_artifact = ModelPusherArtifact(
                model_file_path=model_file_path,
                saved_model_path=saved_model_path
            )

            # push artifact 
            s3_bucket_url = f"s3://{MODEL_PUSHER_S3_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_syncer.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=s3_bucket_url)

            # push best model
            s3_bucket_url = f"s3://{MODEL_PUSHER_S3_BUCKET_NAME}/best_model/{self.training_pipeline_config.timestamp}/model.joblib"
            self.s3_syncer.upload_file_to_s3(file_path=model_pusher_artifact.saved_model_path, aws_bucket_url=s3_bucket_url)

            return model_pusher_artifact
        
        except Exception as e:
            raise TelcoChurnMLException(e, sys)