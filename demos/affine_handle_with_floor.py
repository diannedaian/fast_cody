import sys
import os

# Add the source directory to the path so we can import directly from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fast_cody.apps.interactive_cd_affine_handle import interactive_cd_affine_handle

interactive_cd_affine_handle()
