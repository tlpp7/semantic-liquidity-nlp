"""Optional S3-compatible artifact uploader. Explicit opt-in; no secrets in repository."""
from pathlib import Path

def upload_artifact(local_path, bucket, key, endpoint_url=None):
    import boto3
    local = Path(local_path)
    if not local.is_file():
        raise FileNotFoundError(local)
    if not bucket or not key or key.startswith("/"):
        raise ValueError("bucket and relative key required")
    client = boto3.client("s3", endpoint_url=endpoint_url)
    client.upload_file(str(local), bucket, key)
    return {"bucket": bucket, "key": key, "bytes": local.stat().st_size}
