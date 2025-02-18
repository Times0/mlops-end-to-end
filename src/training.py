import os
import mlflow
from mlflow import MlflowClient
from ultralytics import YOLO
from dotenv import load_dotenv
from rich.console import Console

console = Console()

# Load environment variables
load_dotenv('src/config/.env')

class Train():
    def __init__(self, model_name):
        self.model_name = model_name
        self.data_yaml = os.path.join(os.getcwd(), os.getenv("DATA_YAML", "data/THE-dataset/yolo.yaml"))
        self.yolo_dir = os.path.join(os.getcwd(),  os.getenv("YOLO_DIR_TMP", "tmp/yolo_runs"))
        if os.getenv("HAS_GPU", "NO")=="YES":
            self.device = 0
        else:
            self.device = "mps"

        os.makedirs(self.yolo_dir, exist_ok=True)


    def train_model(self, epochs):
        with mlflow.start_run(run_name=self.model_name, log_system_metrics=True) as run:

            model = YOLO(model=self.model_name+".pt")

            model.train(
                data=self.data_yaml,
                epochs=epochs,
                device= self.device,
                project=self.yolo_dir,  #dir to save runs
                name="train",      #name of this run
                exist_ok=True,      #overwrite existing files
                close_mosaic=0,
                seed=42
            )
                
            #upload weights
            mlflow.log_artifact(
                local_path = os.path.join(self.yolo_dir, "train/weights/best.pt"),
                artifact_path = "weights",
                run_id=run.info.run_id,
            )

            mlflow.log_artifact(
                local_path="requirements.txt",
                artifact_path="environment",
                run_id=run.info.run_id,
            )


    def register_model(self):

        run_id = mlflow.last_active_run().info.run_id

        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/weights/best.pt", name=self.model_name
        )

        client = MlflowClient()

        champion_version = None
        try: #Try to get the champion version
            champion_version = client.get_model_version_by_alias(self.model_name, "Champion")
            champion_run = mlflow.get_run(champion_version.run_id)
        except:
            console.print("Couldn't get champion. Current model will be crowned")
            pass

        if champion_version is None:
            client.set_registered_model_alias(
                name=self.model_name, alias="Champion", version=model_version.version
            )
        else:
            KEY_METRIC = "metrics/mAP50-95B"
            current_metrics = mlflow.get_run(run_id).data.metrics
            champion_metrics = mlflow.get_run(champion_version.run_id).data.metrics

            console.print(f"current : {current_metrics}\nchampion : {champion_metrics}")
            if current_metrics.get(KEY_METRIC, 0) > champion_metrics.get(KEY_METRIC, 0):
                console.print("[green]Current model is better than champion: setting alias Challenger[/]")
                client.set_registered_model_alias(
                    name=self.model_name, alias="Challenger", version=model_version.version
                )
            else:
                console.print("[green]Current model is worse than champion: not setting alias[/]")


if __name__ == "__main__":
    epochs = 2 #in env
    train = Train("yolo11n")
    train.train_model(epochs)
    train.register_model()
