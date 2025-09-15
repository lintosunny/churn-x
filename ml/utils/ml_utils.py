import os 
import sys 
import yaml
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from ml.entity.artifact_entity import ClassificationMetricArtifact
from ml.logger import logging
from ml.exception import TelcoChurnException

def classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        model_fl_score = f1_score(y_true, y_pred)
        model_accuracy_score = accuracy_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_recall_score_1 = recall_score(y_true, y_pred, average=None, pos_label=1)
        model_recall_score_0 = recall_score(y_true, y_pred, average=None, pos_label=0)
        model_precision_score = precision_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_fl_score,
            accuracy_score=model_accuracy_score,
            recall_score=model_fl_score,
            recall_score_1=model_recall_score_1,
            recall_score_0=model_recall_score_0,
            precision_score=model_precision_score
        )

        return classification_metric
    except Exception as e:
        raise TelcoChurnException(e, sys)
    

def Te