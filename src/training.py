import os
from ultralytics import YOLO
from pathlib import Path
import mlflow
from ultralytics import settings
import boto3
from os import getenv

from rich.console import Console

from dotenv import load_dotenv

## IMPORTANT ##
# Start infrastructure before using `docker-compose up`

# Update a setting
settings.update({"mlflow": True})

console = Console()

EPOCHS = 12
DATASET_PATH = "data/THE-dataset"


def setup():
    mlflow.set_tracking_uri("http://localhost:5000")
    
    #load_dotenv()

    # Configure MLflow to use MinIO - use Docker service name since MLflow runs in Docker
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"  # For MLflow in Docker
    os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin"

        # When setting up MLflow
    artifact_root = getenv('MLFLOW_DEFAULT_ARTIFACT_ROOT') #artifact_root = getenv('MLFLOW_DEFAULT_ARTIFACT_ROOT', 's3://mlflow/mlflow-artifacts/')
    mlflow.set_experiment_tag("artifact_location", artifact_root)
    mlflow.set_experiment("YOLO_Training")

#TODO load from env
def setup_s3_client():
    s3_client = boto3.client(
        's3',
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )
    try:
        s3_client.head_bucket(Bucket='mlflow')
    except:
        s3_client.create_bucket(Bucket='mlflow')

    """
    boto3.client(
            's3',
            endpoint_url="http://localhost:9000",  # For direct S3 access from host
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            aws_session_token=None,
            config=boto3.session.Config(signature_version='s3v4'),
            verify=False
        )
    """

if __name__ == "__main__":
    console.print("[green]Starting training...[/]")
    # MLflow UI is exposed to host on port 5000

    setup()
    # Create the mlflow bucket in MinIO if it doesn't exist - use localhost since we're on host
    setup_s3_client()


    cwd = Path.cwd()
    path = cwd / "data" / "THE-dataset"


    with mlflow.start_run() as run:
        # Log a simple test metric
        mlflow.log_metric("test_metric", 1.0)
        
        model = YOLO("yolo11n")
        
        # Train the model
        results = model.train(
            data=os.path.join(path, "yolo.yaml"),
            epochs=EPOCHS
        )
        
        # Log model
        mlflow.pytorch.log_model(model.model, "model")
        
        # Get metrics after they've been logged
        current_metrics = mlflow.get_run(run.info.run_id).data.metrics

        client = mlflow.MlflowClient()

        console.print("[green]Checking if challenger is better than champion...[/]")
        #champion_model = mlflow.pytorch.load_model("models:/Champion/latest")
        #champion_model = mlflow.pytorch.load_model("models:/YOLO-model/Champion") ##

        champion_version = None
        try:
            champion_version = client.get_model_version_by_alias("YOLO-model", "Champion")
            console.print(f"[blue]Champion version details:[/]")
            console.print(f"Version: {champion_version.version}")
            console.print(f"Run ID: {champion_version.run_id}")
            console.print(f"Current stage: {champion_version.current_stage}")
            
            champion_run = mlflow.get_run(champion_version.run_id)
            console.print(f"[blue]Champion run details:[/]")
            console.print(f"Run ID: {champion_run.info.run_id}")
            console.print(f"Status: {champion_run.info.status}")
            console.print(f"Artifact URI: {champion_run.info.artifact_uri}")
            console.print(f"Metrics: {champion_run.data.metrics}")

            # After getting champion run
            champion_run = mlflow.get_run(champion_version.run_id)
            console.print("[blue]Full champion run data:[/]")
            console.print(f"Info: {champion_run.info}")
            console.print(f"Data: {champion_run.data}")
            console.print(f"Tags: {champion_run.data.tags}")
        except:
            pass


        #THERE IS A CHAMPION
        if champion_version is not None:
            console.print(f"[blue]Champion version: {champion_version}[/]")
            champion_run_id = champion_version.run_id
            console.print(f"[blue]Champion run_id: {champion_run_id}[/]")
            #uri = f"runs:/{champion_run_id}/weights/best.pt"
            champion_metrics = mlflow.get_run(champion_run_id).data.metrics

        #download_uri = mlflow.artifacts.download_artifacts(uri)

            console.print(f"current : {current_metrics}\nchampion : {champion_metrics}")

            KEY_METRIC = "metrics/mAP50-95B"

            console.print(f"{KEY_METRIC}\ncurrent : {current_metrics.get(KEY_METRIC, 0)}\nchampion : {champion_metrics.get(KEY_METRIC, 0)}")
            # Comparer les performances (exemple avec la métrique map50-95)
            if current_metrics.get(KEY_METRIC, 0) > champion_metrics.get(
                KEY_METRIC, 0
                ):
                model_version = mlflow.register_model(
                    f"runs:/{run.info.run_id}/model",
                    "YOLO-model",
                )
                client.set_registered_model_alias(
                    "YOLO-model", "Challenger", model_version.version
                )
                console.print("[green]Better than champion[/]")
            else:
                mlflow.register_model(
                    f"runs:/{run.info.run_id}/model", "YOLO-model"
                )
                console.print("[green]Worse than champion[/]")

        #THERE IS NO CHAMPION
        else:
            console.print("[green]No champion model found, registering as champion...[/]")
            model_version = mlflow.register_model(
                f"runs:/{run.info.run_id}/model",
                "YOLO-model",
            )
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                "YOLO-model", "Champion", model_version.version
            )
            console.print("[green]Champion model registered successfully![/]")


