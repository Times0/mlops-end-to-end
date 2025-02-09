# Getting started

https://app.picsellia.com/

Create a `.env` file at the root of the project with the provided environment variables.

Starting

```
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

Launch the MLflow infrastructure on another terminal:
```
docker-compose up
```
(or with '-d' to run in background)

Now run the scripts:
```
uv run src/dataset_preparation.py # download dataset / prepare dataset for training
uv run src/training.py # start training with ultralytics
```

The MLflow UI will be available at http://localhost:5001
MinIO console will be available at http://localhost:9001
