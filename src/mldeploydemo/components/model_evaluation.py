import os
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        actual = np.array(actual).ravel()
        pred = np.array(pred).ravel()

        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        # 1. Load test data & model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        # 2. Make script use EXACTLY the same MLflow config as notebook
        #    (hardcoding here just to prove it works; later you can move to config/env.)
        os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/nisinha77/e_2_e_ml.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "nisinha77"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "52722fd8203edb2c5e8513115301eb6f69eae5cf"

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # just like notebook
        registered_model_name = "ElasticnetModel"
        run_name = "model_evaluation_from_script"

        # Sanity print: this should match notebook
        print("SCRIPT tracking URI:", mlflow.get_tracking_uri())
        print("SCRIPT registry URI:", mlflow.get_registry_uri())
        print("SCRIPT MLFLOW_TRACKING_USERNAME:", os.environ.get("MLFLOW_TRACKING_USERNAME"))

        with mlflow.start_run(run_name=run_name):
            # 3. Predict & compute metrics
            predicted_qualities = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # 4. Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # 5. Log params & metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # 6. Log + REGISTER model — same as in notebook
            #    (DagsHub supports the MLflow API, so this should create/update ElasticnetModel versions.)
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=registered_model_name,
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                )
