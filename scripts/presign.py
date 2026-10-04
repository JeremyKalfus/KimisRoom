"""Presigned S3 URLs for the confirmatory box, which holds no AWS credentials of its own.

    uv run --no-project --with boto3 python scripts/presign.py BUCKET SECONDS get|put KEY [KEY ...]

Prints one URL per key. Credentials come from the environment (scripts/confirm_run.sh runs under
one `kv exec`); URLs signed with a long-term key last at most 7 days.
"""

import sys

import boto3
from botocore.config import Config

bucket, seconds, method, *keys = sys.argv[1:]
# the regional endpoint: a new bucket answers the global one with a 307 for a while, and curl -f lets a 307 through
s3 = boto3.client("s3", region_name="us-east-2", endpoint_url="https://s3.us-east-2.amazonaws.com",
                  config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
op = {"get": "get_object", "put": "put_object"}[method]
for k in keys:
    print(s3.generate_presigned_url(op, Params={"Bucket": bucket, "Key": k}, ExpiresIn=int(seconds)))
