"""
Demo script for simple floor viewer.
"""
import sys
import os

# Add the source directory to the path so we can import directly from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fast_cody.apps.simple_floor_viewer import simple_floor_viewer

if __name__ == "__main__":
    simple_floor_viewer()

