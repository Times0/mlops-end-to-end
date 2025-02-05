# Getting started

https://app.picsellia.com/

Create a `.env` file at the root of the project with the following content:
```
PICSELIA_API_TOKEN = "your-api-token"
```

Starting
```
uv venv
uv pip install -r requirements.txt
cd src
python dataset_preparation.py # download dataset / prepare dataset for training
python training.py # start training with ultralytics
```
