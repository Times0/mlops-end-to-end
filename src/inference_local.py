import os
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
import tempfile
import requests

# Set up the MLflow client
client = MlflowClient()

# Get the registered model by name
model_name = "yolo11n"
alias = "Champion"

# Get the model version by alias
model_version = client.get_model_version_by_alias(name=model_name, alias=alias)

# Get the run ID from the model version
run_id = model_version.run_id

# Get the artifact URI directly
artifact_uri = client.get_run(run_id).info.artifact_uri
model_path = f"{artifact_uri}/weights/best.pt"

# Remove 's3://' prefix if present
if model_path.startswith('s3://'):
    model_path = model_path[5:]

# Construct the full MinIO URL
minio_url = f"{os.getenv('MLFLOW_S3_ENDPOINT_URL')}/{model_path}"
print(f"Attempting to load model from: {minio_url}")

# Download the model file to a temporary location
temp_dir = tempfile.mkdtemp()
local_model_path = os.path.join(temp_dir, "best.pt")

response = requests.get(minio_url)
with open(local_model_path, 'wb') as f:
    f.write(response.content)

# Load the model from the local file
model = YOLO(local_model_path)

def infer(inference_type=None, source=None):
    if inference_type=="image":
        if source is None:
            print("No image. Using default")
            source="yolotest.jpeg"  
        results = model.predict(source, show=True)
    elif inference_type=="video":
        if source is None:
            source="https://youtu.be/LNwODJXcvt4"
        results = model.track(source, show=True) #Can lead to ram filling up in slower systems.
    elif inference_type=="camera":
        results = model.track("0", show=True)
    else:
        print("inference type must be \"camera\", \"image\" or \"video\"")

infer("image")