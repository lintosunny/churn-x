import joblib
import os 
from mcp_server.cloud.s3_syncer import S3Sync
from mcp_server.constants import MODEL_PATH, S3_BUCKET_NAME


s3_syncer = S3Sync()

def load_model():
    """
    Downloads the latest churn model from S3 and loads it.
    """
    # Ensure the directory exists
    os.makedirs(MODEL_PATH, exist_ok=True)

    # only download if no model inside
    if not os.listdir(MODEL_PATH):
        base_s3_url = f"s3://{S3_BUCKET_NAME}/best_model"
        s3_bucket_url = s3_syncer.get_latest_model_s3_path()
        s3_syncer.sync_folder_from_s3(s3_bucket_url, MODEL_PATH)

    # Load model
    model = joblib.load(MODEL_PATH)
    return model


