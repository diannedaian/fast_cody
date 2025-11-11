# -*- coding: utf-8 -*-
"""
fast_cody.apps.pipeline
---------------------------------
End-to-end pipeline: Image → Hunyuan GLB → OBJ + Texture → MSH

Takes an input image, generates a 3D model via Hunyuan API,
converts it to OBJ format with textures, and meshes it with TetWild.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict

# Import directly to avoid loading full fast_cody package dependencies
import importlib.util
spec_hunyuan = importlib.util.spec_from_file_location("hunyuan", Path(__file__).parent / "hunyuan.py")
hunyuan_module = importlib.util.module_from_spec(spec_hunyuan)
spec_hunyuan.loader.exec_module(hunyuan_module)
hunyuan_generate = hunyuan_module.hunyuan_generate

spec_convert = importlib.util.spec_from_file_location("convert_to_msh", Path(__file__).parent / "convert_to_msh.py")
convert_module = importlib.util.module_from_spec(spec_convert)
spec_convert.loader.exec_module(convert_module)
convert_glb_to_msh = convert_module.convert_glb_to_msh


def run_pipeline(
    image_path: str,
    hunyuan_out: str = "outputs/hunyuan",
    converted_out: str = "outputs/converted",
    tetwild_path: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Run the complete Fast Cody 3D generation pipeline.

    Args:
        image_path: Path to input image file.
        hunyuan_out: Output directory for Hunyuan GLB files.
        converted_out: Output directory for converted OBJ/MSH files.
        tetwild_path: Optional path to TetWild executable.

    Returns:
        dict: Dictionary with paths to generated files:
            {
                "glb": glb_path,
                "obj": obj_path,
                "texture": texture_path,
                "msh": msh_path
            }
    """
    print("\n" + "=" * 50)
    print("===== FAST CODY 3D PIPELINE START =====")
    print("=" * 50 + "\n")

    # Step 1: Generate GLB from image using Hunyuan
    print("[1] Submitting image to Hunyuan 2D→3D ...")
    try:
        glb_path = hunyuan_generate(image_path, hunyuan_out)
        if not glb_path:
            print("[ERROR] Hunyuan generation failed. Aborting pipeline.")
            return {
                "glb": None,
                "obj": None,
                "texture": None,
                "msh": None,
            }

        glb_path = Path(glb_path).resolve()
        print(f"[1] ✅ GLB generated: {glb_path}\n")

    except Exception as e:
        print(f"[ERROR] Hunyuan generation failed: {e}")
        return {
            "glb": None,
            "obj": None,
            "texture": None,
            "msh": None,
        }

    # Step 2: Convert to MSH (skip GLB→OBJ if already OBJ)
    glb_path_str = str(glb_path)
    try:
        if glb_path_str.endswith(".obj"):
            print("[2] Model is already OBJ format, converting to MSH ...")
            # OBJ already exists, just need to convert to MSH
            obj_stem = Path(glb_path_str).stem
            out_root = Path(converted_out).resolve()
            job_dir = out_root / obj_stem
            job_dir.mkdir(parents=True, exist_ok=True)

            # Copy OBJ and texture to converted directory
            import shutil
            obj_path = job_dir / "model.obj"
            shutil.copy2(glb_path_str, obj_path)

            # Check for texture in same directory as OBJ
            texture_src = Path(glb_path_str).parent / "texture.png"
            if texture_src.exists():
                texture_path = job_dir / "texture.png"
                shutil.copy2(texture_src, texture_path)
                print(f"[CONVERT] Copied texture to: {texture_path}")

            # Convert OBJ to MSH
            msh_path = convert_module.convert_to_msh(str(obj_path), job_dir, tetwild_path)
        else:
            print("[2] Converting GLB to OBJ + Texture + MSH ...")
            msh_path = convert_glb_to_msh(glb_path_str, converted_out, tetwild_path)
        msh_path = Path(msh_path).resolve()

        # Infer OBJ and texture paths from MSH path
        # MSH is at: {converted_out}/{glb_stem}/model.msh
        # OBJ is at: {converted_out}/{glb_stem}/model.obj
        # Texture is at: {converted_out}/{glb_stem}/texture.png
        msh_dir = msh_path.parent
        obj_path = msh_dir / "model.obj"
        texture_path = msh_dir / "texture.png"

        # Check if files actually exist
        if not obj_path.exists():
            print(f"[WARNING] OBJ file not found at expected location: {obj_path}")
            obj_path = None
        else:
            obj_path = obj_path.resolve()

        if not texture_path.exists():
            print(f"[INFO] No texture file found (this is normal if GLB had no texture)")
            texture_path = None
        else:
            texture_path = texture_path.resolve()

        print(f"[2] ✅ Conversion complete: {msh_path}\n")

    except Exception as e:
        print(f"[ERROR] GLB conversion failed: {e}")
        return {
            "glb": str(glb_path),
            "obj": None,
            "texture": None,
            "msh": None,
        }

    # Step 3: Summary
    print("=" * 50)
    print("✅ Pipeline complete!")
    print("=" * 50)
    print(f"GLB:  {glb_path}")
    if obj_path:
        print(f"OBJ:  {obj_path}")
    if texture_path:
        print(f"PNG:  {texture_path}")
    if msh_path:
        print(f"MSH:  {msh_path}")
    print("=" * 50 + "\n")

    result = {
        "glb": str(glb_path),
        "obj": str(obj_path) if obj_path else None,
        "texture": str(texture_path) if texture_path else None,
        "msh": str(msh_path) if msh_path else None,
    }

    # Save summary JSON
    try:
        summary_path = msh_dir / "pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"📄 Summary saved to: {summary_path}\n")
    except Exception as e:
        print(f"[WARNING] Failed to save summary JSON: {e}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fast Cody 3D Pipeline: Image → GLB → OBJ → MSH"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image file",
    )
    parser.add_argument(
        "--tetwild",
        default=None,
        help="Path to TetWild executable (auto-detected if not provided)",
    )
    parser.add_argument(
        "--hunyuan-out",
        default="outputs/hunyuan",
        help="Output directory for Hunyuan GLB files (default: outputs/hunyuan)",
    )
    parser.add_argument(
        "--converted-out",
        default="outputs/converted",
        help="Output directory for converted OBJ/MSH files (default: outputs/converted)",
    )
    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"[ERROR] Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    result = run_pipeline(
        str(image_path),
        args.hunyuan_out,
        args.converted_out,
        args.tetwild,
    )

    # Exit with appropriate code
    if result["msh"]:
        print("✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("❌ Pipeline failed. Check error messages above.", file=sys.stderr)
        sys.exit(1)
