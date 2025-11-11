# -*- coding: utf-8 -*-
"""
fast_cody.apps.hunyuan
---------------------------------
Step 1 of the pipeline:

Takes an image, calls Tencent Cloud Hunyuan 2D-to-3D API,
downloads the resulting model (.obj or .glb) and saves it
under outputs/hunyuan/<timestamp>_<jobid>/.
"""

import os
import json
import time
import requests
import base64
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ai3d.v20250513 import ai3d_client, models
from typing import Optional

# Load environment variables from .env file
load_dotenv()


def hunyuan_generate(
    image_path: str, out_dir: str = "outputs/hunyuan"
) -> str:
    """
    Generate a 3D model from an input image using Tencent Hunyuan API.

    Args:
        image_path (str): Path to the input image (local file or URL).
        out_dir (str): Base directory for saving output.

    Returns:
        str: Absolute path to the downloaded model file (.obj or .glb).
    """
    # 1. Ensure output directory exists
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        # 2. Load credentials
        # WARNING: Hardcoded credentials are a security risk. Use environment variables in production.
        secret_id = os.getenv("TENCENTCLOUD_SECRET_ID")
        secret_key = os.getenv("TENCENTCLOUD_SECRET_KEY")
        if not secret_id or not secret_key:
            raise EnvironmentError(
                "Missing TENCENTCLOUD_SECRET_ID or TENCENTCLOUD_SECRET_KEY environment variables."
            )

        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile(endpoint="ai3d.tencentcloudapi.com")
        client_profile = ClientProfile(httpProfile=http_profile)
        # Region is required - trying common regions for AI3D service
        # Common regions: ap-beijing, ap-shanghai, ap-guangzhou, ap-chengdu
        # Try ap-guangzhou as it's often used for newer services
        region = "ap-guangzhou"
        client = ai3d_client.Ai3dClient(cred, region, client_profile)

        # 3. Prepare request
        req = models.SubmitHunyuanTo3DJobRequest()
        params = {}

        # Handle local files vs URLs
        if image_path.startswith("http"):
            params["ImageUrl"] = image_path
        else:
            # For local files, encode as base64
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            with open(image_file, "rb") as f:
                image_data = f.read()
            params["ImageBase64"] = base64.b64encode(image_data).decode('utf-8')

        req.from_json_string(json.dumps(params))

        # 4. Submit job
        resp = client.SubmitHunyuanTo3DJob(req)
        job_info = json.loads(resp.to_json_string())
        job_id = job_info.get("JobId") or job_info.get("Job", {}).get("JobId")
        if not job_id:
            raise RuntimeError("Failed to obtain job_id from response.")
        print(f"[HUNYUAN] Job submitted successfully: {job_id}")

        # 5. Poll job status until completion
        status_req = models.QueryHunyuanTo3DJobRequest()
        while True:
            status_req.JobId = job_id
            status_resp = client.QueryHunyuanTo3DJob(status_req)
            status_data = json.loads(status_resp.to_json_string())

            # Handle different response structures
            state = status_data.get("Status") or status_data.get("Job", {}).get("Status") or "UNKNOWN"
            print(f"[HUNYUAN] Job {job_id} status: {state}")

            if state in ["DONE", "SUCCEEDED", "FAILED"]:
                break
            time.sleep(5)

        if state in ["FAILED"]:
            error_msg = status_data.get("ErrorMessage") or status_data.get("Error", {}).get("Message", "")
            raise RuntimeError(f"Hunyuan job {job_id} failed: {error_msg}")

        # 6. Get output URLs - handle different response structures
        # New API structure: ResultFile3Ds array
        result_files = status_data.get("ResultFile3Ds", [])
        if result_files:
            # Get the first result file (usually OBJ or GLB)
            model_url = result_files[0].get("Url")
            texture_url = None  # Texture might be in the ZIP or separate
        else:
            # Old API structure: Job.Output
            outputs = status_data.get("Job", {}).get("Output", {})
            model_url = outputs.get("ModelUrl")
            texture_url = outputs.get("TextureUrl")

        # 7. Create unique save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = out_root / f"{timestamp}_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        def download(url, filename):
            if not url:
                return None
            response = requests.get(url, stream=True)
            response.raise_for_status()
            save_path = job_dir / filename
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return save_path

        # 8. Download model and texture
        if not model_url:
            raise RuntimeError("No model URL found in job response")

        # Download the file first to check its type
        print(f"[HUNYUAN] Downloading model from: {model_url[:80]}...")
        temp_path = download(model_url, "model_temp")
        if not temp_path:
            raise RuntimeError("Failed to download model file")

        # Check if it's a ZIP file by trying to open it
        import zipfile
        is_zip = False
        try:
            with zipfile.ZipFile(temp_path, 'r') as test_zip:
                is_zip = True
        except (zipfile.BadZipFile, zipfile.LargeZipFile):
            is_zip = False

        if is_zip:
            print(f"[HUNYUAN] Model is in ZIP format, extracting...")
            zip_dir = job_dir / "extracted"
            zip_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall(zip_dir)
            # Look for OBJ and texture files in extracted directory
            obj_files = list(zip_dir.glob("*.obj"))
            png_files = list(zip_dir.glob("*.png"))
            if obj_files:
                # Copy OBJ to main directory
                import shutil
                model_path = job_dir / "model.obj"
                shutil.copy2(obj_files[0], model_path)
                print(f"[HUNYUAN] Extracted OBJ: {model_path}")
            else:
                # Try GLB if no OBJ
                glb_files = list(zip_dir.glob("*.glb"))
                if glb_files:
                    model_path = job_dir / "model.glb"
                    shutil.copy2(glb_files[0], model_path)
                    print(f"[HUNYUAN] Extracted GLB: {model_path}")
                else:
                    raise RuntimeError("No OBJ or GLB file found in ZIP")
            if png_files:
                texture_path = job_dir / "texture.png"
                shutil.copy2(png_files[0], texture_path)
                print(f"[HUNYUAN] Extracted texture: {texture_path}")
            # Clean up temp file
            temp_path.unlink()
        else:
            # Not a ZIP, rename to appropriate extension
            if model_url.endswith(".glb") or ".glb" in model_url.lower():
                model_path = job_dir / "model.glb"
            elif model_url.endswith(".obj") or ".obj" in model_url.lower():
                model_path = job_dir / "model.obj"
            else:
                # Try to detect from content
                model_path = job_dir / "model.glb"
            temp_path.rename(model_path)
            print(f"[HUNYUAN] Model saved as: {model_path}")

            if texture_url:
                download(texture_url, "texture.png")

        # 9. Save metadata
        with open(job_dir / "metadata.json", "w") as f:
            json.dump(status_data, f, indent=2)

        print(f"[HUNYUAN] Model saved at {model_path}")
        return str(model_path.resolve())

    except TencentCloudSDKException as err:
        print("[HUNYUAN] Tencent SDK Error:", err)
    except Exception as e:
        print("[HUNYUAN] Error:", e)

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Hunyuan 2D→3D model generation")
    parser.add_argument("--image", required=True, help="Path or URL to input image")
    parser.add_argument("--out", default="outputs/hunyuan", help="Output directory")
    args = parser.parse_args()

    result = hunyuan_generate(args.image, args.out)
    if result:
        print("✅ Model generated:", result)
    else:
        print("❌ Model generation failed.")
