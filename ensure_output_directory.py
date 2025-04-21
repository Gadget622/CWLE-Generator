#!/usr/bin/env python3

"""
Helper script to ensure the output directory structure exists
"""

import os
import sys

def ensure_output_directory():
    """
    Create the output directory if it doesn't exist
    """
    output_dir = ".output"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            sys.exit(1)
    else:
        print(f"Output directory already exists: {output_dir}")

if __name__ == "__main__":
    ensure_output_directory()