# 🎯 Object Detection Fine-tuning Pipeline

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

To stop the services:
```shell
docker compose down
```

💡 Tip: Use `-v` flag with `down` command to remove volumes for a fresh start

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

To serve models with BentoML (static image inference only):
```shell
bentoml build
bentoml serve
```

Test the deployment using `inference_bento` (remember to update the endpoint URL).

## 📝 TODO
- [ ] Refactor code
- [ ] Implement automatic S3 bucket creation
- [ ] Add docstrings
- [*] Clean up README
