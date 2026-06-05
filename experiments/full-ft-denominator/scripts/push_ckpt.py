#!/usr/bin/env python3
"""Upload a checkpoint directory to Wasabi (S3-compatible) with multipart.

Usage: push_ckpt.py <local_dir> <dataset> [seed]
  -> uploads every file under <local_dir> to
     s3://$WASABI_BUCKET/seed<seed>/<dataset>/<relpath>

Env: WASABI_ACCESS_KEY, WASABI_SECRET_KEY, WASABI_ENDPOINT (default s3.wasabisys.com),
     WASABI_BUCKET (default peftbench-fullft-ckpts).
Skips files already present in the bucket with the same size (idempotent re-runs).
"""
import os, sys, boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

local_dir = sys.argv[1]
dataset   = sys.argv[2]
seed      = sys.argv[3] if len(sys.argv) > 3 else "42"

BUCKET   = os.environ.get("WASABI_BUCKET", "peftbench-fullft-ckpts")
ENDPOINT = os.environ.get("WASABI_ENDPOINT", "https://s3.wasabisys.com")
ak = os.environ["WASABI_ACCESS_KEY"]; sk = os.environ["WASABI_SECRET_KEY"]

s3 = boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=ak,
                  aws_secret_access_key=sk, config=Config(signature_version="s3v4"))
# 256 MB parts, 8-way concurrency — good for 16-100 GB shards over a fast link.
xfer = TransferConfig(multipart_threshold=256*1024*1024, multipart_chunksize=256*1024*1024,
                      max_concurrency=8, use_threads=True)

prefix = f"seed{seed}/{dataset}"
# map existing objects -> size for idempotent skip
existing = {}
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix + "/"):
    for o in page.get("Contents", []):
        existing[o["Key"]] = o["Size"]

uploaded = skipped = 0
total_bytes = 0
for root, _, files in os.walk(local_dir):
    for fn in files:
        fp = os.path.join(root, fn)
        rel = os.path.relpath(fp, local_dir)
        key = f"{prefix}/{rel}"
        sz = os.path.getsize(fp)
        if existing.get(key) == sz:
            skipped += 1
            continue
        s3.upload_file(fp, BUCKET, key, Config=xfer)
        uploaded += 1; total_bytes += sz
        print(f"  up {key} ({sz/1e9:.2f} GB)", flush=True)
print(f"PUSH DONE {dataset}: {uploaded} up / {skipped} skip / {total_bytes/1e9:.1f} GB -> s3://{BUCKET}/{prefix}/", flush=True)
