## Object detection fine-tuning pipeline.
This project was done in the context of a class in MLOps imparted by [@picsalex].
In this project, we propose a pipeline for training object detection models fine-tuned on a dataset of small items (Granola, Purple Balisto, Tuna tin, Plastic Bottle, Bueno White, Bueno Black, Chocolate bar, Kinder Délice, Snikers, and Twix). 

The data is hosted via Picsellia, the model registry and metadata store is done via mlflow, which uses mysql and minio for the metadata and the artifacts respectfully. Serving of the model is done with Bentoml, or locally. The local version can use data from images, videos or the camera for inferences, whilst the bentoml is constrained to images.


To run this project, install uv and docker.


### Preparation
Set up the python environment
```
uv venv --python=3.11
uv pip install -r requirements.txt 
source .venv/bin/activate
```

Now you must set the src/config/.env file. Copy .env.distrib and set the missing variables


We use docker to compartementalize the pieces of the pipeline.
Check if docker is running
```
systemctl status docker
```
If docker is not running, start it with:
```
systemctl start docker
```
start the run in background
```
docker compose up --build -d
```
(You can turn down docker with the following command once you are done)
```
docker compose down
```
If you want to delete the volumes for a fresh start, use "-v"


### Running
run dataset_preparation, training and inference_local in that order
Configuration is done throught the src/config/.env file. 


If you have vscode, you can launch them in the run section.
If you are using a terminal, you must overload the MLFLOW_S3_ENDPOINT_URL environment variable like this:
```
MLFLOW_S3_ENDPOING_URL=http://localhost:9000 uv run src/<file.py>
```

To serve on bentoml (only for static image inference, not for video or camera use):
```
bentoml cloud login
bentoml build
bentoml serve
```

You can then use the inference_bento (update the endpoint url) to test it.








TODO:
docstring & type hints
clean readme