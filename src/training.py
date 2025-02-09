import os
from ultralytics import YOLO
from pathlib import Path
import mlflow
from ultralytics import settings

## IMPORTANT ##
# Start mlflow before using `mlflow server --host 0.0.0.0 --port 5000`

# Update a setting
settings.update({"mlflow": True})

EPOCHS = 4
DATASET_PATH = "data/THE-dataset"


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("YOLO_Training")

    cwd = Path.cwd()
    path = cwd / "data" / "THE-dataset"

    with mlflow.start_run():
        model = YOLO("yolo11n")
        model.train(
            data=os.path.join(path, "yolo.yaml"),
            epochs=EPOCHS,
        )

        mlflow.pytorch.log_model(model.model, "model")
        current_metrics = mlflow.get_run(mlflow.active_run().info.run_id).data.metrics

        # Check if challenger is better than champion
        try:
            champion_model = mlflow.pytorch.load_model("models:/Champion/latest")
            champion_run = mlflow.get_run(
                mlflow.pytorch.get_model_info("models:/Champion/latest").run_id
            )
            champion_metrics = champion_run.data.metrics

            # Comparer les performances (exemple avec la métrique map50-95)
            if current_metrics.get("metrics/map50-95", 0) > champion_metrics.get(
                "metrics/map50-95", 0
            ):
                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/model",
                    "YOLO-model",
                    tags={"status": "Challenger"},
                )
            else:
                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/model", "YOLO-model"
                )
        except mlflow.exceptions.MlflowException:
            # Si pas de Champion, enregistrer comme Champion
            model_version = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                "YOLO-model",
                tags={"status": "Champion"},
            )
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                "YOLO-model", "Champion", model_version.version
            )
