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

from ml.utils.ml_utils import classification_score
from ml.utils.ml_utils.estimator import TelcoChurnModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    # ---------------- Optuna objective function: Logistic Regression ----------------
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
            scorer = make_scorer(recall_score, average='binary', pos_label=1)
            return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()
        return objective

    # ---------------- Optuna objective function: Random Forest ----------------
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
            scorer = make_scorer(recall_score, average='binary', pos_label=1)
            return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()
        return objective
    
    # ---------------- Optuna objective function: K Nearest Neighbors ----------------
    def create_objective_knn(self, X_train, y_train):
        """Create objective function for Random Forest with datasets"""
        def objective(trial):
            model = KNeighborsClassifier(
                n_neighbors=trial.suggest_int("n_neighbors", 3, 15),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                metric=trial.suggest_categorical("metric", ["euclidean", "manhattan"])
            )
            scorer = make_scorer(recall_score, average='binary', pos_label=1)
            return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()
        return objective
    
    # ---------------- Optuna objective function: XGBoost ----------------
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
            scorer = make_scorer(recall_score, average='binary', pos_label=1)
            return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()
        return objective

    def perform_hyper_parameter_tuning(self, X_train, y_train, X_test, y_test, n_trials=100):
        try:
            logging.info("Starting hyperparameter tuning for all models...")
            
            # Create objective functions with datasets
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
                
                study = optuna.create_study(direction="maximize")
                study.optimize(objective_func, n_trials=n_trials)
                logging.info(f"{model_name} best CV recall score: {study.best_value:.4f}")
                logging.info(f"{model_name} best params: {study.best_params}")

                # Create and train model with best parameters
                if model_name == "LogisticRegression":
                    best_model = LogisticRegression(**study.best_params, random_state=42)
                elif model_name == "RandomForestClassifier":
                    best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
                elif model_name == "KNeighborsClassifier":
                    best_model = KNeighborsClassifier(**study.best_params)
                elif model_name == "XGBoostXGBClassifier":
                    best_model = XGBClassifier(**study.best_params)
                
                # Fit model and get test score
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                test_recall_1 = recall_score(y_test, y_pred, average='binary', pos_label=1)

                # Store results
                model_results[model_name] = {
                    "best_params": study.best_params,
                    "cv_score": study.best_value,
                    "test_score": test_recall_1,
                    "trained_model": best_model
                }

                # Track best overall model
                if test_recall_1 > best_overall_score:
                    best_overall_score = test_recall_1
                    best_overall_model = best_model
                    best_model_name = model_name

                logging.info(f"{model_name} - CV Score: {study.best_value:.4f}, Test Score: {test_recall_1:.4f}")

            # Save results to JSON (excluding the trained model objects)
            results_for_json = {}
            for name, result in model_results.items():
                results_for_json[name] = {
                    "best_params": result["best_params"],
                    "cv_score": result["cv_score"],
                    "test_score": result["test_score"]
                }
            
            write_json_file(self.model_trainer_config.hyper_parameter_optuna_file_path, results_for_json)

            logging.info(f"Best overall model: {best_model_name} with test recall score: {best_overall_score:.4f}")
            
            return best_overall_model, best_model_name, model_results

        except Exception as e:
            raise TelcoChurnException(e, sys)

    def initiate_model_training(self):
        """Main method to initiate model training"""
        try:
            # Load your datasets
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
            
            # Call hyperparameter tuning
            best_model, best_model_name, all_results = self.perform_hyper_parameter_tuning(
                x_train, y_train, x_test, y_test, n_trials=5
            )
            
            logging.info(f"Training completed. Best model: {best_model_name}")

            # get best model metrics
            y_train_pred = best_model.predict(x_train)
            classification_train_metric = classification_score(y_train_pred, y_train)
            if classification_train_metric.recall_score_1 <= self.model_trainer_config.expected_recall_1:
                raise Exception("Trained model is not good to provide expected accuracy")
            
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = classification_report(y_test_pred, y_test)

            # overfiting and underfiting
            diff = abs(classification_train_metric.recall_score_1 - classification_test_metric.recall_score_1)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is underfitting or overfitting. Do more experimentation")
            
            preprocessor = load_object(file_path=data_transformation_artifact.transformed_file_object)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            telco_churn_model = 


            return best_model, all_results
            
        except Exception as e:
            raise TelcoChurnException(e, sys)