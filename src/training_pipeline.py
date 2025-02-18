from pathlib import Path
import mlflow
from mlflow import MlflowClient
from ultralytics import YOLO
from dotenv import load_dotenv
from rich.console import Console
import boto3
from config import config

load_dotenv()
KEY_METRIC = "metrics/mAP50-95B"

console = Console()


class Trainer:
    def __init__(
        self,
        model_name: str,
        data_yaml: str | Path,
        yolo_dir: str | Path,
        device: str | int,
    ) -> None:
        """
        Initializes the trainer wraper.

        Args:
            model_name (str): The name of the model (e.g. yolo11n).
            data_yaml (str, Path): Path to the data yaml configuration file.
            yolo_dir (str, Path): Directory where the logs and model will be temporarily stored (overwriting past runs).
            device (str, int): The device to run the model on (e.g. cpu, cuda:0, cuda:1, etc.). Either the device name or a number indicating which GPU to use
        """
        self.model_name = model_name
        self.data_yaml = Path(data_yaml)
        self.yolo_dir = Path(yolo_dir)
        self.device = device

        s3_client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        )
        try:
            s3_client.head_bucket(Bucket="mlflow")
        except Exception:
            s3_client.create_bucket(Bucket="mlflow")

    def train_model(self) -> None:
        """
        Trains the model configured in __init__.
        Logs the results in yolo_dir (unused; you can delete this freely) and in mlflow for latter retrieval.
        """
        with mlflow.start_run(run_name=self.model_name, log_system_metrics=True) as run:
            model = YOLO(model=self.model_name + ".pt")

            # Training + Validation done automatically by ultralytics
            model.train(
                data=str(self.data_yaml),
                epochs=config.EPOCHS,
                device=self.device,
                project=str(self.yolo_dir),  # dir to save runs
                close_mosaic=0,
                seed=42,
            )

            # upload weights
            mlflow.log_artifact(
                local_path=self.yolo_dir / "train/weights/best.pt",
                artifact_path="weights",
                run_id=run.info.run_id,
            )

            mlflow.log_artifact(
                local_path="requirements.txt",
                artifact_path="environment",
                run_id=run.info.run_id,
            )

    def register_model(self) -> None:
        """
        Registers the model and assigns an alias.
        The alias can either be Champion if there is no earlier run of the model or Challenger if it performs better than the current Champion.
        An alias is unique in mlflow. The challenger alias refers to the last challenger. All the challenger models are identified via the "status" tag
        """
        run_id = mlflow.last_active_run().info.run_id
        model_version = mlflow.register_model(model_uri=f"runs:/{run_id}/weights/best.pt", name=self.model_name)
        client = MlflowClient()

        champion_version = None
        try:  # Try to get the champion version
            champion_version = client.get_model_version_by_alias(self.model_name, "Champion")
        except Exception:
            console.print("Couldn't get champion. Current model will be crowned")
            pass

        if champion_version is None:
            client.set_registered_model_alias(name=self.model_name, alias="Champion", version=model_version.version)
            client.set_registered_model_tag(name=self.model_name, key="status", value="Champion")
        else:
            current_metrics = mlflow.get_run(run_id).data.metrics
            champion_metrics = mlflow.get_run(champion_version.run_id).data.metrics

            console.print(f"current : {current_metrics}\nchampion : {champion_metrics}")
            if current_metrics.get(KEY_METRIC, 0) > champion_metrics.get(KEY_METRIC, 0):
                console.print("[green]Current model is better than champion: setting alias Challenger[/]")
                client.set_registered_model_alias(
                    name=self.model_name,
                    alias="Challenger",
                    version=model_version.version,
                )
                client.set_registered_model_tag(name=self.model_name, key="status", value="Challenger")
            else:
                console.print("[green]Current model is worse than champion: not setting alias[/]")


def main():
    trainer = Trainer(
        model_name=config.MODEL_NAME,
        data_yaml=Path.cwd() / config.DATA_YAML,
        yolo_dir=Path.cwd() / config.YOLO_DIR_TMP,
        device=config.DEVICE,
    )
    # PIPELINE ML 4 & 5 : Training + Evaluation
    trainer.train_model()

    # PIPELINE ML 6 : Model Validation
    trainer.register_model()


if __name__ == "__main__":
    main()
