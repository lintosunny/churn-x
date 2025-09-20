import os
import subprocess

class S3Sync:
    def sync_folder_to_s3(self,folder,aws_bucket_url):
        command = f"aws s3 sync {folder} {aws_bucket_url} "
        os.system(command)

    def sync_folder_from_s3(self,folder,aws_bucket_url):
        command = f"aws s3 sync  {aws_bucket_url} {folder} "
        os.system(command)

    def get_latest_model_s3_path(self, base_s3_url: str) -> str:
        """
        Get latest timestamp folder from S3 (CLI-based).
        Example: base_s3_url = s3://bucket/best_model/
        """
        command = ["aws", "s3", "ls", base_s3_url]
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # Extract subfolders (timestamps)
        folders = [
            line.split()[-1].strip("/")
            for line in result.stdout.splitlines()
            if line.endswith("/")
        ]
        if not folders:
            raise ValueError(f"No model folders found in {base_s3_url}")

        latest = sorted(folders)[-1]
        return f"{base_s3_url}{latest}/"
    