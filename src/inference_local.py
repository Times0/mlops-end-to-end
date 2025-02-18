"""
Predicts the bounding boxes of the relevant objects, using a model previously trained.
Inference is done locally in contrast with inference_bento.py which does remote inference and then outputs the result.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
from minio import Minio
import boto3
import shutil

from config_inference import *

#Paths
tmp_dir = os.path.join(os.getcwd(), "tmp")
models_dir = os.path.join(tmp_dir, "models")
local_model_path = os.path.join(models_dir, "best.pt")

test_files_dir = os.path.join(os.getcwd(), "test_files") #Where the sample images are located
default_image_path = os.path.join(test_files_dir, "image.jpeg")
inference_result_path = 'results_inference.jpg'
#default_video_path = os.path.join(test_files_dir, "<VIDEO>") #No default video


class Inference():
    def __init__(self, model_name, model_alias):
        self.model_name = model_name
        self.model_alias = model_alias

        self.client = MlflowClient()
        self.model_version = self.client.get_model_version_by_alias(name=self.model_name, alias=self.model_alias)
        self.run_id = self.model_version.run_id    
        self.model = None

        self.overwrite_model = True

        #Setup paths and files
        #os.rmdir(tmp_dir)
        shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)


    def load_model(self):
        """
        Loads model from minio using the uri given by mlflow
        """
        
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


    def infer(self):
        """
        Does a prediction on the inputs given via config_interface.py
        """

        def infer_image(source):
            print(f"Image absolute path: {os.path.abspath(source)}")
            if not os.path.exists(source): 
                print("Image provided does not exist")
                exit(0)

            results = self.model.predict(source, show=True)

            for r in results:
                im_array = r.plot()
                import cv2
                cv2.imwrite(inference_result_path, im_array)

        def infer_video(source):
            results = self.model.track(source, show=True) #Can lead to ram filling up in slower systems.



        source = None

        if source_type_inference =="image":
            print("Running image inference")
            try:
                source = file_inference
            except:
                print("No image. Using default")
                source = default_image_path
                
            print(source)
            infer_image(source)

        elif source_type_inference == "video":
            print("Running video inference")
            try:
                source = file_inference
            except:
                #source="https://youtu.be/LNwODJXcvt4" #That's a video from ultralytics that won't work well with the finetuned model.
                print("Provide a video to track. It can be a file, but also a link.")
                exit(0)
            infer_video(source)

        elif source_type_inference=="camera":
            results = self.model.track("0", show=True)

        else:
            print("Inference type must be \"camera\", \"image\" or \"video\"")

if __name__=="__main__":

    inference = Inference(
        model_name = "yolo11n",
        model_alias = "Champion",
    )
    inference.load_model()
    inference.infer() #Change in config_inference.py