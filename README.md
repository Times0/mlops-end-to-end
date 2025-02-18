# 🎯 Object Detection Fine-tuning Pipeline
This project was done in the context of a class in MLOps imparted by [@picsalex].
In this project, we propose a pipeline for training object detection models fine-tuned on a dataset of small items (Granola, Purple Balisto, Tuna tin, Plastic Bottle, Bueno White, Bueno Black, Chocolate bar, Kinder Délice, Snikers, and Twix). 

The data is hosted via Picsellia, the model registry and metadata store is done via mlflow, which uses mysql and minio for the metadata and the artifacts respectfully. Serving of the model is done with Bentoml, or locally. The local version can use data from images, videos or the camera for inferences, whilst the bentoml is constrained to images.

A Docker-based pipeline for object detection model fine-tuning and deployment.

## 🚀 Getting Started

### Prerequisites
- Docker 🐳

### 🛠️ Environment Setup

1. Set up the Python environment:
```shell
pip install uv
uv venv --python=3.11
uv pip install -r requirements.txt
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Now you must set the src/config/.env file. Copy .env.distrib and set the missing variables


We use docker to compartementalize the pieces of the pipeline.
Check if docker is running
```

2. 🐳 Docker Setup
Check Docker status:
```shell
systemctl status docker
```
If Docker isn't running:
```shell
systemctl start docker
```

3. 🚀 Launch the Pipeline
Start services in background:
```shell
docker compose up --build -d
```
(You can turn down docker with the following command once you are done)
To stop the services:
```shell
docker compose down
```

Tip: Use `-v` flag with `down` command to remove volumes for a fresh start

## 🏃‍♂️ Running the Pipeline
Follow these steps in order:
1. Dataset Preparation
2. Training
3. Local Inference

### 🖥️ Running Options

#### VSCode
Launch directly from the Run section in VSCode

#### Terminal
Set the MLFLOW endpoint URL before running:

Linux:
```shell
MLFLOW_S3_ENDPOING_URL=http://localhost:9000 uv run src/<file.py>
```

Windows:
```powershell
$env:MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"; uv run src/<file.py>
```

### 🚀 Deployment

To serve on bentoml (only for static image inference, not for video or camera use):
```
bentoml cloud login
bentoml build
bentoml serve
```

Test the deployment using `inference_bento` (remember to update the endpoint URL).

## 📝 TODO
- [ ] Refactor code
- [ ] Implement automatic S3 bucket creation
- [ ] docstrings and typehints everywhere ?
- [x] Clean up README
