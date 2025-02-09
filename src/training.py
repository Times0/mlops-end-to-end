import os
from ultralytics import YOLO
from pathlib import Path
import mlflow
from ultralytics import settings
import boto3

## IMPORTANT ##
# Start infrastructure before using `docker-compose up`

# Update a setting
settings.update({"mlflow": True})

EPOCHS = 4
DATASET_PATH = "data/THE-dataset"

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Configure MLflow to use MinIO
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin"
    
    # Create the mlflow bucket in MinIO if it doesn't exist
    s3_client = boto3.client(
        's3',
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        aws_session_token=None,
        config=boto3.session.Config(signature_version='s3v4'),
        verify=False
    )
    
    try:
        s3_client.head_bucket(Bucket='mlflow')
    except:
        s3_client.create_bucket(Bucket='mlflow')
    
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
