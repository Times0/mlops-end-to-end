import logging
import os
from os import access
import mlflow
from mlflow.tracking import MlflowClient
from minio import Minio

import bentoml
import numpy as np
from PIL import Image

logger = logging.getLogger("bentoml")
logger.setLevel(logging.DEBUG)

tmp_dir = os.path.join(os.getcwd(), "tmp")
models_dir = os.path.join(tmp_dir, "models")
local_model_path = os.path.join(models_dir, "best.pt")

@bentoml.service
class YOLOService:
    def __init__(self) -> None:
        self.model_name = "yolo11n",
        self.model_alias = "Champion",
        self.model = self.load_model("yolo11n.pt")

        self.overwrite_model = True #If there is a model already downloaded from mlflow, overwrite it

        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

    def load_model(self)->None:
        """Load the YOLO model using MLflow"""
        from ultralytics import YOLO    

        client = MlflowClient()
        model_version = client.get_model_version_by_alias(name=self.model_name, alias=self.model_alias)
        run_id = model_version.run_id
        artifact_uri = client.get_run(run_id).info.artifact_uri
        model_path = f"{artifact_uri}/weights/best.pt"  

        if model_path.startswith('s3://'): 
            model_path = model_path[5:]
        bucket_name, object_name = model_path.split('/', 1)
        
        endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        endpoint = endpoint.removeprefix("http://").removeprefix("https://")
        minio_client = Minio(
            endpoint,
            access_key=os.getenv('AWS_ACCESS_KEY_ID'),
            secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            secure=False
        )
        
        if (not os.path.exists(local_model_path)) or (self.overwrite_model == True):
            minio_client.fget_object(bucket_name, object_name, local_model_path)
            print(f"Model downloaded to: {local_model_path}")
        else:
            print(f"Using existing model at: {local_model_path}")

        self.model = YOLO(local_model_path)

    @bentoml.api
    async def predict(self, image: Image.Image) -> dict:
        """
        Handle prediction requests

        Args:
            image: PIL Image object

        Returns:
            dict: Prediction results including boxes, scores, and class labels
        """
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run inference
        results = self.model.predict(img_array)
        result = results[0]  # Get first result since we only process one image

        # Format response
        boxes = []
        for box in result.boxes:
            boxes.append(
                {
                    "xyxy": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                }
            )

        return {"boxes": boxes, "inference_time": float(result.speed["inference"])}


# curl -X POST "https://yolo-service-ast1.mt-guc1.bentoml.ai/predict" -F "image=@/Users/alexis/Downloads/image.jpg
