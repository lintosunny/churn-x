import os
import sys 
import optuna
import dagshub
import mlflow
import joblib
from ml.exception import TelcoChurnException
from ml.logger import logging
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score,make_scorer
from ml.utils.main_utils import (
    load_numpy_array_data, 
    read_json_file, save_object, 
    load_object, 
    write_json_file
)
from ml.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from ml.entity.config_entity import ModelTrainerConfig
from ml.utils.ml_utils.metric import classification_score
from ml.utils.ml_utils.estimator import TelcoChurnModel

load_dotenv(override=True)


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    # -------------------------------------------------------------------------
    # Objective Functions for Optuna (one per algorithm)
    # -------------------------------------------------------------------------
    def create_objective_lr(self, X_train, y_train):
        """Create objective function for Logistic Regression with datasets"""
        def objective(trial):
            model = LogisticRegression(
                C=trial.suggest_float('C', 1e-4, 1e4, log=True),
                solver=trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga', 'newton-cg']),
                tol=trial.suggest_float('tol', 1e-6, 1e-1, log=True),
                class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
                random_state=42
            )
            scorer = make_scorer(recall_score, average="binary", pos_label=1)
            return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()
        return objective

    def create_objective_rf(self, X_train, y_train):
        """Create objective function for Random Forest with datasets"""
        def objective(trial):
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 5, 30),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                random_state=42,
                n_jobs=-1
            )
            scorer = make_scorer(recall_score, average="binary", pos_label=1)
            return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()
        return objective
    
    def create_objective_knn(self, X_train, y_train):
        """Create objective function for Random Forest with datasets"""
        def objective(trial):
            model = KNeighborsClassifier(
                n_neighbors=trial.suggest_int("n_neighbors", 3, 15),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                metric=trial.suggest_categorical("metric", ["euclidean", "manhattan"])
            )
            scorer = make_scorer(recall_score, average="binary", pos_label=1)
            return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()
        return objective
    
    def create_objective_xgb(self, X_train, y_train):
        """Create objective function for Random Forest with datasets"""
        def objective(trial):
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
            scorer = make_scorer(recall_score, average="binary", pos_label=1)
            return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()
        return objective

    # -------------------------------------------------------------------------
    # Hyperparameter tuning with Optuna
    # -------------------------------------------------------------------------
    def perform_hyper_parameter_tuning(self, X_train, y_train, X_test, y_test, n_trials=50):
        try:
            logging.info("Starting hyperparameter tuning for all models...")
            
            # Model objectives
            objectives = {
                "LogisticRegression": self.create_objective_lr(X_train, y_train),
                "RandomForestClassifier": self.create_objective_rf(X_train, y_train),
                "KNeighborsClassifier": self.create_objective_knn(X_train, y_train),
                "XGBoostXGBClassifier": self.create_objective_xgb(X_train, y_train)
            }

            model_results = {}
            best_overall_model = None
            best_overall_score = 0
            best_model_name = ""

            for model_name, objective_func in objectives.items():
                logging.info(f"Tuning {model_name}...")
                
                # Run Optuna
                study = optuna.create_study(direction="maximize")
                study.optimize(objective_func, n_trials=n_trials)
                logging.info(f"{model_name} best CV recall score: {study.best_value:.4f}")
                logging.info(f"{model_name} best params: {study.best_params}")

                # Re-train best model with best params
                if model_name == "LogisticRegression":
                    best_model = LogisticRegression(**study.best_params, random_state=42)
                elif model_name == "RandomForestClassifier":
                    best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
                elif model_name == "KNeighborsClassifier":
                    best_model = KNeighborsClassifier(**study.best_params)
                elif model_name == "XGBoostXGBClassifier":
                    best_model = XGBClassifier(**study.best_params)
                
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                test_recall_1 = recall_score(y_test, y_pred, average="micro")

                # Save results
                model_results[model_name] = {
                    "best_params": study.best_params,
                    "cv_score": study.best_value,
                    "test_score": test_recall_1
                }

                # Track best overall model
                if test_recall_1 > best_overall_score:
                    best_overall_score = test_recall_1
                    best_overall_model = best_model
                    best_model_name = model_name

                logging.info(f"{model_name} - CV Score: {study.best_value:.4f}, Test Score: {test_recall_1:.4f}")

            # Save tuning results
            results_for_json = {}
            for name, result in model_results.items():
                results_for_json[name] = {
                    "best_params": result["best_params"],
                    "cv_score": result["cv_score"],
                    "test_score": result["test_score"]
                }
            
            write_json_file(self.model_trainer_config.hyper_parameter_optuna_file_path, results_for_json)

            logging.info(f"Best overall model: {best_model_name} with test recall score: {best_overall_score:.4f}")
            
            return best_model_name, best_overall_model

        except Exception as e:
            raise TelcoChurnException(e, sys)
        
    # -------------------------------------------------------------------------
    # MLflow Tracking
    # -------------------------------------------------------------------------
    def track_mlflow(self, best_model, classification_test_metric, classification_train_metric):
        mlflow.set_tracking_uri(os.getenv("MLFLOW TRACKING URI"))
        mlflow.set_experiment("Telco_Churn_Prediction")

        with mlflow.start_run():
            # Log train metrics
            mlflow.log_metric("train_f1_score", classification_train_metric.f1_score)
            mlflow.log_metric("train_accuracy_score", classification_train_metric.accuracy_score)
            mlflow.log_metric("train_precision_score", classification_train_metric.precision_score)
            mlflow.log_metric("train_recall_score", classification_train_metric.recall_score)
            mlflow.log_metric("train_recall_score_1", classification_train_metric.recall_score_1)
            mlflow.log_metric("train_recall_score_0", classification_train_metric.recall_score_0)

            # Log test metrics
            mlflow.log_metric("test_f1_score", classification_test_metric.f1_score)
            mlflow.log_metric("test_accuracy_score", classification_test_metric.accuracy_score)
            mlflow.log_metric("test_precision_score", classification_test_metric.precision_score)
            mlflow.log_metric("test_recall_score", classification_test_metric.recall_score)
            mlflow.log_metric("test_recall_score_1", classification_test_metric.recall_score_1)
            mlflow.log_metric("test_recall_score_0", classification_test_metric.recall_score_0)

            logging.info("✅ Train & Test metrics saved successfully")


            joblib.dump(best_model, "model.joblib")
            mlflow.log_artifact("model.joblib", artifact_path="model")
            logging.info("✅ Model saved as artifact successfully")

            # Cleanup local files
            os.remove("model.joblib")

    # -------------------------------------------------------------------------
    # Main Training Pipeline
    # -------------------------------------------------------------------------
    def initiate_model_training(self):
        """Orchestrates the entire training workflow."""
        try:
            # Load train & test arrays
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            logging.info("📂 Data loaded successfully for training & testing")

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            
            # Step 1: Hyperparameter tuning
            best_model_name, best_model = self.perform_hyper_parameter_tuning(
                x_train, y_train, x_test, y_test
            )
            logging.info(f"Training completed. Best model: {best_model_name}")

            # Step 2: Evaluate on train & test
            y_train_pred = best_model.predict(x_train)
            classification_train_metric = classification_score(y_train_pred, y_train)
            if classification_train_metric.recall_score_1 <= self.model_trainer_config.expected_recall_1:
                raise Exception("Trained model is not good to provide expected accuracy")
            
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = classification_score(y_test_pred, y_test)

            # Step 3: Check for overfitting/underfitting
            diff = abs(classification_train_metric.recall_score_1 - classification_test_metric.recall_score_1)
            logging.info(f"Train: {classification_train_metric.recall_score_1}, Test: {classification_test_metric.recall_score_1}, Difference: {diff}")
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is underfitting or overfitting. Do more experimentation")
            
            # Step 4: MLflow tracking
            self.track_mlflow(best_model, classification_test_metric, classification_train_metric)
            
            # Step 5: Save final TelcoChurnModel with preprocessor
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            telco_churn_model = TelcoChurnModel(preprocessor=preprocessor, model=best_model)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, obj=telco_churn_model)

            # Step 6: Return Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                train_metric_artifact = classification_train_metric,
                test_metric_artifact = classification_test_metric,
                is_model_accepted = None,
                improved_accuracy = None,
                best_model_path = None,
                best_model_metric_artifact = None
            )

            logging.info("Model training pipeline completed successfully")
            return model_trainer_artifact
            
        except Exception as e:
            raise TelcoChurnException(e, sys)