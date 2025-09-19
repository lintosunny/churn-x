import os 
import sys 
import pandas as pd 

from ml.logger import logging 
from ml.exception import TelcoChurnMLException

from ml.entity.config_entity import ModelEvaluationConfig
from ml.entity.artifact_entity import (
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)

from ml.utils.ml_utils.metric import classification_score
from ml.utils.ml_utils.estimator import TelcoChurnModel, ModelResolver
from ml.constants.training_pipeline import TARGET_COLUMN
from ml.utils.main_utils import save_object, load_object, write_yaml_file



class ModelEvaluation:
    def __init__(
            self, 
            model_evaluation_config: ModelEvaluationConfig, 
            data_validation_artifact: DataValidationArtifact, 
            model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise TelcoChurnMLException(e, sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path 

            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            df = pd.concat([train_df, test_df])
            y_true = df[TARGET_COLUMN]
            df.drop(TARGET_COLUMN, axis=1, inplace=True)

            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted = True

            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_path=self.model_trainer_artifact.train_metric_artifact,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.train_metric_artifact,
                    best_model_metric_artifact=self.model_trainer_artifact.train_metric_artifact
                )

                model_eval_report = model_evaluation_artifact.__dict__
                write_yaml_file(self.model_evaluation_config.report_file_path, model_eval_report)
                logging.info("This is the first model. So saving as best model")
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            lastest_model_file_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=lastest_model_file_path)
            train_model = load_object(file_path=train_model_file_path)

            y_trained_pred = train_model.predict(df)
            y_latest_pred = train_model.predict(df)

            trained_metric = classification_score(y_true, y_trained_pred)
            latest_metric = classification_score(y_true, y_latest_pred)

            improved_accuracy = trained_metric.recall_score_1 - latest_metric.recall_score_1
            if self.model_evaluation_config.change_threshold < improved_accuracy:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=improved_accuracy,
                    best_model_path=train_model_file_path,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=trained_metric,
                    best_model_metric_artifact=trained_metric
                )

                model_eval_report = model_evaluation_artifact.__dict__
                write_yaml_file(self.model_evaluation_config.report_file_path, model_eval_report)
                logging.info("Replacing trained model with best model")
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            
            else:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=False,
                    improved_accuracy=None,
                    best_model_path=lastest_model_file_path,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=trained_metric,
                    best_model_metric_artifact=latest_metric
                )

                model_eval_report = model_evaluation_artifact.__dict__
                write_yaml_file(self.model_evaluation_config.report_file_path, model_eval_report)
                logging.info("Latest model is the best model. No changes made.")
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

        except Exception as e:
            raise TelcoChurnMLException(e, sys)

