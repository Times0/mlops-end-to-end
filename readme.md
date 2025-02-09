# Getting started

https://app.picsellia.com/

Create a `.env` file at the root of the project with the following content:
```
PICSELIA_API_TOKEN = "your-api-token"
```

Starting

```
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

launch mlflow server
On another terminal:
```
source .venv/bin/activate
mlflow server --host 0.0.0.0 --port 5000
```

Now run the scripts:
```
uv run src/dataset_preparation.py # download dataset / prepare dataset for training
uv run src/training.py # start training with ultralytics
```
