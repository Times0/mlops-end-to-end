import os
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

source="yolotest3.jpeg" 
model_path = "src/yolo_runs/train/weights/best.pt"
model = YOLO(model_path)
results = model.predict(source, show=True)
