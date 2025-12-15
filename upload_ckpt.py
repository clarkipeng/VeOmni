
import os
import boto3
from pathlib import Path
from loguru import logger
import dotenv

dotenv.load_dotenv()

LOCAL_PATH = "./qwen3_vl_moe_sft_2e-5/checkpoints/global_step_465/hf_ckpt_converted"
S3_BUCKET = "camfer-weights"
S3_PREFIX = "gcd-run-42/a3b_moe/qwen3_vl_moe_sft_2e-5"

def upload_checkpoint_to_s3(local_path: str, bucket: str, prefix: str) -> None:
    s3_client = boto3.client("s3")
    local_path_obj = Path(local_path)
    
    if not local_path_obj.exists():
        logger.error(f"Local checkpoint path does not exist: {local_path}")
        return
    
    # Upload all files in the checkpoint directory
    uploaded_count = 0
    files = list(local_path_obj.rglob("*"))
    print(f"Found {len([f for f in files if f.is_file()])} files to upload...")
    
    for file_path in files:
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path_obj)
            s3_key = f"{prefix}/{relative_path}".replace("\\", "/") 
            
            print(f"Uploading {relative_path} to s3://{bucket}/{s3_key}...")
            s3_client.upload_file(str(file_path), bucket, s3_key)
            uploaded_count += 1
    
    print(f"Uploaded {uploaded_count} files from {local_path} to s3://{bucket}/{prefix}")

if __name__ == "__main__":
    upload_checkpoint_to_s3(LOCAL_PATH, S3_BUCKET, S3_PREFIX)
