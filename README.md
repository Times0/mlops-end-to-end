uv venv --python=3.11
uv pip install -r requirements.txt 
source .venv/bin/activate

(systemctl status docker
systemctl start docker

docker compose up --build -d

then run dataset_preparation, training and inference_local in that order, using vscode or overloading the MLFLOW_S3_ENDPOINT_URL like this:

MLFLOW_S3_ENDPOING_URL=http://localhost:9000 uv run src/training.py 

To serve on bentoml (only static image inference):
bentoml build
bentoml serve

You can then use the inference_bento (update the endpoint url) to test it.

TODO:
refactor
create bucket automatically on s3
docstring
clean readme
report