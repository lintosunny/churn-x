import os 
import sys
import subprocess
from mcp_server.logger import logging
from mcp_server.exception import TelcoChurnMCPException

def sync_folder_from_s3(folder, aws_bucket_url):
    """
    Syncs a local folder with an AWS S3 bucket

    Args:
        folder (str): Path to the local folder where files will be synced
        aws_bucket_url (str): Full S3 bucket URL (e.g., s3://my-bucket/path)
    """
    command = ["aws", "s3", "sync", aws_bucket_url, folder]

    logging.info(f"Starting S3 sync from {aws_bucket_url} to {folder}...")

    try:
        # Run the command and capture output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info("s3 sync completed succesfully")
        if result.stdout:
            logging.debug(f"Command Output:\n{result.stdout}")

    except Exception as e:
        raise TelcoChurnMCPException(e, sys) from e