import bentoml
from ultralytics import YOLO

client = MlflowClient()
artifact_uri = self.client.get_run(self.run_id).info.artifact_uri
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
    secure=False  #true if https
)

if (not os.path.exists(local_model_path)) or (self.overwrite_model == True):
    minio_client.fget_object(bucket_name, object_name, local_model_path)
    print(f"Model downloaded to: {local_model_path}")
else:
    print(f"Using existing model at: {local_model_path}")

self.model = YOLO(local_model_path)

model = YOLO(local_path)

bentoml.pytorch.save_model("yolo_model", model)
