import os
import dotenv

# Load .env file and check if it was successful
if not dotenv.load_dotenv("src/config/.env"):
    raise RuntimeError("Failed to load .env file")


class Config:
    def __init__(self):
        # Get environment variables with fallback error if not set
        self.api_token = os.environ.get("PICSELIA_API_TOKEN")
        if not self.api_token:
            raise ValueError("PICSELIA_API_TOKEN environment variable is not set")

        self.ORG_NAME = "Picsalex-MLOps"
        self.HOST = "https://app.picsellia.com"  # If host changes, we need to redo everython so no need for env var
        self.DATASET_ID = "0194d124-1c5f-7d01-a532-be8aeebf59e8"


# why use a class?
# We can have autocompletion and type checking 👍

config = Config()
