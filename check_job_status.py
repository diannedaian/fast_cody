#!/usr/bin/env python3
"""Check status of a Hunyuan job"""

import os
import json
from dotenv import load_dotenv
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ai3d.v20250513 import ai3d_client, models

# Load environment variables from .env file
load_dotenv()

# Load credentials
secret_id = os.getenv("TENCENTCLOUD_SECRET_ID")
secret_key = os.getenv("TENCENTCLOUD_SECRET_KEY")

if not secret_id or not secret_key:
    raise EnvironmentError(
        "Missing TENCENTCLOUD_SECRET_ID or TENCENTCLOUD_SECRET_KEY environment variables. "
        "Please set them in your .env file or environment."
    )

cred = credential.Credential(secret_id, secret_key)
http_profile = HttpProfile(endpoint="ai3d.tencentcloudapi.com")
client_profile = ClientProfile(httpProfile=http_profile)
region = "ap-guangzhou"
client = ai3d_client.Ai3dClient(cred, region, client_profile)

# Job ID from previous run
job_id = "1379282349585735680"

print(f"Checking status of job: {job_id}")

try:
    status_req = models.QueryHunyuanTo3DJobRequest()
    status_req.JobId = job_id
    status_resp = client.QueryHunyuanTo3DJob(status_req)
    status_data = json.loads(status_resp.to_json_string())

    print("\n=== Job Status ===")
    print(json.dumps(status_data, indent=2, ensure_ascii=False))

    job_info = status_data.get("Job", {})
    state = job_info.get("Status", "UNKNOWN")
    print(f"\nStatus: {state}")

    if state == "SUCCEEDED":
        outputs = job_info.get("Output", {})
        model_url = outputs.get("ModelUrl")
        texture_url = outputs.get("TextureUrl")
        print(f"Model URL: {model_url}")
        print(f"Texture URL: {texture_url}")
    elif state == "FAILED":
        error = job_info.get("Error", {})
        print(f"Error: {error}")

except Exception as e:
    print(f"Error checking job: {e}")
