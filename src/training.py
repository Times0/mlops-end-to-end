import os
import mlflow
from mlflow import MlflowClient
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv('src/config/.env')

def train_model(model, datayaml, epochs):
    # Get YOLO directory from environment variable with fallback
    yolo_dir = os.path.join(os.getcwd(), os.getenv('YOLO_RUNS_DIR', 'tmp/yolo_runs'))
    os.makedirs(yolo_dir, exist_ok=True)

    with mlflow.start_run(run_name=model, log_system_metrics=True) as run:
        model = YOLO(model=model+".pt")

        model.train(
            data=os.path.join(os.getcwd(), datayaml),
            epochs=epochs,
            device=0, #"mps"
            project=yolo_dir,  # Directory to save runs
            name="train",      # Name of this run
            exist_ok=True      # Overwrite existing files
        )
            
        mlflow.log_artifact(
            local_path=os.path.join(yolo_dir, "train", "weights", "best.pt"),
            artifact_path="weights",
            run_id=run.info.run_id,
        )

        mlflow.log_artifact(
            local_path="requirements.txt",
            artifact_path="environment",
            run_id=run.info.run_id,
        )


def register_model(model):
    run_id = mlflow.last_active_run().info.run_id
    model_name = model

    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/weights/best.pt", name=model_name
    )

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name, alias="Champion", version=model_version.version
    )


if __name__ == "__main__":
    model = "yolo11n"   #in env
    datayaml = "data/THE-dataset/yolo.yaml"  #in env
    epochs = 10 #in env
    train_model(model, datayaml, epochs)
    register_model(model)
