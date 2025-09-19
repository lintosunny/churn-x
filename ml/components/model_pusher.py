from ml.exception import TelcoChurnMLException
from ml.logger import logging
from ml.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from ml.entity.config_entity import ModelPusherConfig
import os 
import sys 
import shutil


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config

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

            return model_pusher_artifact
        
        except Exception as e:
            raise TelcoChurnMLException(e, sys)
