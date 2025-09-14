from ml.utils.main_utils import load_numpy_array_data, read_json_file, save_object, load_object, write_json_file
from ml.exception import TelcoChurnException
from ml.logger import logging
import os
import sys 
# from ml-> metric->classification score
# sensor model to predict
from ml.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from ml.entity.config_entity import ModelTrainerConfig

# --- Machine Learning & Models ---
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# --- Model selection & hyperparameter tuning ---
from sklearn.model_selection import cross_val_score
import optuna
#optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna info logs

# --- Metrics ---
from sklearn.metrics import (
    mean_squared_error,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, make_scorer
)

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    def perform_hyper_parameter_tuning(self, X_train, y_train, X_test, y_test, n_trials=100):
        try:
            # store data in self so objectives can use them
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

            logging.info("Starting hyperparameter tuning for all models...")
            models = {
                "LogisticRegression": self._objective_lr,
                "RandomForestClassifier": self._objective_rf,
                "KNeighborsClassifier": self._objective_knn,
                "XGBoostXGBClassifier": self._objective_xgb
            }

            model_recall_1 = {}
            for model, objective in models.items():
                logging.info(f"Tuning {model}...")
                study = f"study_{model}"
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials)
                logging.info(f"{model} best CV recall score: {study.best_value}")
                logging.info(f"{model} best params: {study.best_params}")

                model_clf = model(**study.best_params)
                model_clf.fit(X_train, y_train)
                y_pred = model_clf.predict(X_test)
                test_recall_1 = recall_score(y_test, y_pred, average='binary', pos_label=1)

                model_recall_1[model] = {
                    "best_params": study.best_params,
                    "cv_score": study.best_value,
                    "test_score": test_recall_1
                }
                write_json_file(self.model_trainer_config.hyper_parameter_optuna_file_path, model_recall_1)

                best_model = ""
                best_test_recall_1 = 0
                best_cv_recall_1 = 0

                if test_recall_1 > best_test_recall_1:
                    best_test_recall_1 = test_recall_1
                    best_model = model_clf

            print(best_model)
            print(model_recall_1)
        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    # ---------------- Objective function ----------------
    def _objective_lr(self, trial):
        model = LogisticRegression(
        C = trial.suggest_float('C', 1e-4, 1e4, log=True),
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga', 'newton-cg']),
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True),
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        )
        scorer = make_scorer(recall_score, average='binary', pos_label=1)
        return cross_val_score(model, self.X_train, self.y_train, cv=5, scoring=scorer, n_jobs=-1).mean()
    
    # ---------------- Objective function ----------------
    def _objective_rf(self, trial):
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=42,
            n_jobs=-1
        )
        scorer = make_scorer(recall_score, average='binary', pos_label=1)
        return cross_val_score(model, self.X_train, self.y_train, cv=5, scoring=scorer, n_jobs=-1).mean()

    # ---------------- Objective function ----------------
    def _objective_knn(self, trial):
        model = KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 15),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical("metric", ["euclidean", "manhattan"])
        )
        scorer = make_scorer(recall_score, average='binary', pos_label=1)
        return cross_val_score(model, self.X_train, self.y_train, cv=5, scoring=scorer, n_jobs=-1).mean()

    # ---------------- Objective function ----------------
    def _objective_xgb(self, trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            gamma=trial.suggest_float("gamma", 0, 5),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        scorer = make_scorer(recall_score, average='binary', pos_label=1)
        return cross_val_score(model, self.X_train, self.y_train, cv=5, scoring=scorer, n_jobs=-1).mean()

    def initiate_model_training(self):
        train_file_path = self.data_transformation_artifact.transformed_train_file_path
        test_file_path = self.data_transformation_artifact.transformed_test_file_path

        train_arr = load_numpy_array_data(train_file_path)
        test_arr = load_numpy_array_data(test_file_path)

        x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
        
        x = self.perform_hyper_parameter_tuning(self, x_train, y_train, x_test, y_test)
        print(x)
        

