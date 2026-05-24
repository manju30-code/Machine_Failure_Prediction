import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, make_scorer
# for model serialization
import joblib
import numpy as np 
import mlflow
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mcfailure-training-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/mkrish2025/Machine-Failure-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/mkrish2025/Machine-Failure-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/mkrish2025/Machine-Failure-Prediction/ytrain.csv"
ytest_path = "hf://datasets/mkrish2025/Machine-Failure-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# Ensure ytrain and ytest are Series for consistent behavior if they come as DataFrames with one column
if isinstance(ytrain, pd.DataFrame) and ytrain.shape[1] == 1:
    ytrain = ytrain.iloc[:, 0]
if isinstance(ytest, pd.DataFrame) and ytest.shape[1] == 1:
    ytest = ytest.iloc[:, 0]

# Calculate the ratio
# 1 is the negative/majority class and 0 is the positive/minority class
# Correctly extract scalar counts for scale_pos_weight
neg_count = (ytrain == 1).sum() # This will now be a scalar
pos_count = (ytrain == 0).sum() # This will now be a scalar
spw_value = neg_count / pos_count

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=spw_value, random_state=42)


# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}


# Model pipeline
model_pipeline = make_pipeline(xgb_model)

import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

with mlflow.start_run():
    # Hyperparameter tuning
    #Giving importance to recall as default grissearch scoring is accuracy
    recall_scorer = make_scorer(recall_score, pos_label=1)

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring=recall_scorer)


    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

# for model serialization

# Save the model locally
model_path = "best_mc_failure_prediction_model_v1.joblib"
joblib.dump(best_model, model_path)

# Log the model artifact
mlflow.log_artifact(model_path, artifact_path="model")
print(f"Model saved as artifact at: {model_path}")

# Upload to Hugging Face
repo_id = "mkrish2025/Machine-Failure-Prediction"
repo_type = "model"

# Step 1: Check if the space exists
api = HfApi()

try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
  print(f"Space '{repo_id}' not found. Creating new space...")
  create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
  print(f"Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj="best_mc_failure_prediction_model_v1.joblib",
    path_in_repo="best_mc_failure_prediction_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
    )
