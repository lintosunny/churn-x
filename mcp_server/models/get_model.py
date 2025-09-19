import joblib
import os 
from mcp_server.utils.s3_loader import sync_folder_from_s3
from mcp_server.constants import MODEL_PATH
from dotenv import load_dotenv

load_dotenv(override=True)


def load_model():
    """
    Downloads the latest churn model from S3 and loads it.
    """
    # Ensure the directory exists
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)

    # Always download latest model
    sync_folder_from_s3(os.getenv("S3_BUCKET"), MODEL_PATH)

    # Load model
    model = joblib.load(MODEL_PATH)
    return model


