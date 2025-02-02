import os
import dotenv

# Load .env file and check if it was successful
if not dotenv.load_dotenv():
    raise RuntimeError("Failed to load .env file")


class Config:
    def __init__(self):
        # Get environment variables with fallback error if not set
        self.api_token = os.environ.get("PICSELIA_API_TOKEN")
        if not self.api_token:
            raise ValueError("PICSELIA_API_TOKEN environment variable is not set")

        self.organization_name = os.environ.get("PICSELIA_ORGANIZATION_NAME")
        if not self.organization_name:
            raise ValueError(
                "PICSELIA_ORGANIZATION_NAME environment variable is not set"
            )

        self.host = "https://app.picsellia.com"  # If host changes, we need to redo everython so no need for env var


# why use a class?
# We can have autocompletion and type checking 👍

config = Config()
