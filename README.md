## Object detection fine-tuning pipeline.

To run this project you need to have docker and docker compose installed.

### Preparation

Set up the python environment

```shell
pip install uv
uv venv --python=3.11
uv pip install -r requirements.txt
source .venv/bin/activate  # On windows, use .venv\Scripts\activate
```

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

You can turn down docker with

```
docker compose down
```

If you want to delete the volumes for a fresh start, use "-v"

### Running

run dataset_preparation, training and inference_local in that order

If you have vscode, you can launch them in the run section.
If you are using a terminal, you must overload the MLFLOW_S3_ENDPOINT_URL environment, that is because the docker uses a different port.

For example, like this on linux

```shell
MLFLOW_S3_ENDPOING_URL=http://localhost:9000 uv run src/<file.py>
```

or like this on windows

```powershell
$env:MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"; uv run src/<file.py>
```

To serve on bentoml (only for static image inference, not for video or camera use):

```shell
bentoml build
bentoml serve
```

You can then use the inference_bento (update the endpoint url) to test it.

TODO:
refactor
create bucket automatically on s3
docstring
clean readme
report
