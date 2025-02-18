import os
import dotenv

# Load .env file and check if it was successful
if not dotenv.load_dotenv():
    raise RuntimeError("Failed to load .env file")


class Config:
    def __init__(self):
        # Get environment variables with fallback error if not set
        self.api_token = os.environ.get("PICSELIA_API_TOKEN")
        if not self.api_token:
            raise ValueError("PICSELIA_API_TOKEN environment variable is not set")

        # Pipeline Data
        self.ORG_NAME = "Picsalex-MLOps"
        self.HOST = "https://app.picsellia.com"  # If host changes, we need to redo everython so no need for env var
        self.DATASET_ID = "0194d124-1c5f-7d01-a532-be8aeebf59e8"

        # Pipeline Training
        self.MODEL_NAME = "yolo11n"
        self.DATA_YAML = "data/THE-dataset/yolo.yaml"
        self.YOLO_DIR_TMP = "tmp/yolo_runs"
        self.DEVICE = "cpu"
        self.EPOCHS = 1

        ## minio
        self.endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")


# why use a class?
# We can have autocompletion and type checking 👍

config = Config()
