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
    def __init__(self, model_name: str, data_yaml: str|os.PathLike, yolo_dir: str|os.PathLike, device: str|int) -> None:
        """
        Initializes the training object.

        Args:
            model_name (str): The name of the model.
            data_yaml (str, os.PathLike): Path to the data yaml configuration file.
            yolo_dir (str, os.PathLike): Directory where the model will be temporarily stored (overwriting past runs).
            device (str, int): The device to run the model on. Either the device name or a number indicating which GPU to use
        """
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.yolo_dir = yolo_dir
        self.device = device

        os.makedirs(self.yolo_dir, exist_ok=True)


    def train_model(self, epochs: int) -> None:
        """
        Trains the model configured in __init__.
        Logs the results in yolo_dir (unused; you can delete this freely) and in mlflow for latter retrieval.

        Args:
            epochs (int): The number of epochs of training.
        """
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


    def register_model(self) -> None:
        """
        Registers the model and assignsn an alias.
        The alias can either be Champion if there is no earlier run of the model or Challenger if it performs better than the current Champion.
        An alias is unique in mlflow. The challenger alias refers to the last challenger. All the challenger models are identified via the "status" tag
        """
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
            client.set_registered_model_tag(
                name=self.model_name, key="status", value="Champion"
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
                client.set_registered_model_tag(
                    name=self.model_name, key="status", value="Challenger"
                )
            else:
                console.print("[green]Current model is worse than champion: not setting alias[/]")


if __name__ == "__main__":

    data_yaml = os.path.join(os.getcwd(), os.getenv("DATA_YAML", "data/THE-dataset/yolo.yaml"))
    yolo_dir = os.path.join(os.getcwd(),  os.getenv("YOLO_DIR_TMP", "tmp/yolo_runs"))
    try:
        device = os.getenv("DEVICE")
    except:
        device = "cpu"
    epochs = int(os.getenv("EPOCHS", 2))
    model_name = os.getenv("MODEL", "yolo11n")

    train = Train(
        model_name=model_name,
        data_yaml=data_yaml,
        yolo_dir=yolo_dir,
        device=device
        )
    
    train.train_model(epochs)
    train.register_model()
