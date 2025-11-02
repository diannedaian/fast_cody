#!/usr/bin/env python3
"""
Helper script to convert OBJ/FBX/STL files to MSH format using TetWild.
This script requires TetWild to be installed separately.

Usage:
    python convert_to_msh.py input.obj output.msh
    python convert_to_msh.py input.fbx output.msh
    python convert_to_msh.py input.stl output.msh
"""

import os
import sys
import subprocess
from pathlib import Path

def find_tetwild():
    """Try to find TetWild executable"""
    # Common locations
    possible_paths = [
        "TetWild",
        "./TetWild/build/TetWild",
        "../TetWild/build/TetWild",
        "tetwild",
        "/usr/local/bin/TetWild",
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

    # Try in PATH
    try:
        result = subprocess.run(['which', 'TetWild'],
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        pass

    return None

def convert_to_msh(input_file, output_file, tetwild_path=None):
    """
    Convert a surface mesh (OBJ, STL, etc.) to MSH format using TetWild.

    Parameters
    ----------
    input_file : str
        Path to input mesh file (OBJ, STL, PLY, etc.)
    output_file : str
        Path to output MSH file
    tetwild_path : str, optional
        Path to TetWild executable. If None, will try to find it automatically.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if tetwild_path is None:
        tetwild_path = find_tetwild()

    if tetwild_path is None:
        print("ERROR: TetWild executable not found!")
        print("\nTo use this script, you need to install TetWild:")
        print("1. Clone: git clone https://github.com/Yixin-Hu/TetWild.git")
        print("2. Build: cd TetWild && mkdir build && cd build && cmake .. && make")
        print("3. Either add to PATH or specify path: python convert_to_msh.py input.obj output.msh --tetwild /path/to/TetWild")
        sys.exit(1)

    if not os.path.exists(tetwild_path):
        raise FileNotFoundError(f"TetWild executable not found: {tetwild_path}")

    print(f"Converting {input_file} to {output_file}...")
    print(f"Using TetWild: {tetwild_path}")

    # Run TetWild
    cmd = [tetwild_path, '-i', input_file, '-o', output_file]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"\n✓ Successfully converted to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Conversion failed!")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert OBJ/FBX/STL files to MSH format using TetWild'
    )
    parser.add_argument('input', help='Input mesh file (OBJ, STL, PLY, etc.)')
    parser.add_argument('output', help='Output MSH file')
    parser.add_argument('--tetwild', help='Path to TetWild executable')

    args = parser.parse_args()

    convert_to_msh(args.input, args.output, args.tetwild)
