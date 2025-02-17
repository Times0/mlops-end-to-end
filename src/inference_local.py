import os
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
from minio import Minio

# Set up the MLflow client
client = MlflowClient()

# Get the registered model by name
model_name = "yolo11n"
alias = "Champion"

# Get the model version by alias
model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
print(model_version)
# Get the run ID from the model version
run_id = model_version.run_id
print(run_id)
# Get the artifact URI directly
artifact_uri = client.get_run(run_id).info.artifact_uri
model_path = f"{artifact_uri}/weights/best.pt"

# Parse the S3 path to get bucket and object name
if model_path.startswith('s3://'):
    model_path = model_path[5:]
bucket_name, object_name = model_path.split('/', 1)

# Set up MinIO client
minio_client = Minio(
    "localhost:9000",
    access_key=os.getenv('AWS_ACCESS_KEY_ID'),
    secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    secure=False  # Set to True if using HTTPS
)

# Create tmp directory and its subdirectories
tmp_dir = os.path.join(os.getcwd(), "tmp")
models_dir = os.path.join(tmp_dir, "models")
runs_dir = os.path.join(tmp_dir, "yolo_runs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)

# Download the model file to the models directory
local_model_path = os.path.join(models_dir, "best.pt")

# Download the file using MinIO client if it doesn't exist
if not os.path.exists(local_model_path):
    minio_client.fget_object(bucket_name, object_name, local_model_path)
    print(f"Model downloaded to: {local_model_path}")
else:
    print(f"Using existing model at: {local_model_path}")


# Load the model from the local file
model = YOLO(local_model_path)

def infer(inference_type=None, source=None):
    if inference_type=="image":
        print("Running image inference")
        if source is None:
            print("No image. Using default")
            source="yolotest3.jpeg"  
        print(f"Attempting to predict on image: {source}")
        print(f"Image exists: {os.path.exists(source)}")
        print(f"Image absolute path: {os.path.abspath(source)}")
        results = model.predict(source, show=True)
        print(f"Prediction results: {results}")
        # Save the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            import cv2
            cv2.imwrite('results.jpg', im_array)
    elif inference_type=="video":
        if source is None:
            source="https://youtu.be/LNwODJXcvt4"
        results = model.track(source, show=True) #Can lead to ram filling up in slower systems.
    elif inference_type=="camera":
        results = model.track("0", show=True)
    else:
        print("inference type must be \"camera\", \"image\" or \"video\"")

infer("image")