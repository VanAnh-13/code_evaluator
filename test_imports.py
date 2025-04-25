"""
Test script to verify that all required packages can be imported
"""

import sys
import importlib

def test_import(package_name):
    try:
        importlib.import_module(package_name)
        print(f"✓ Successfully imported {package_name}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {package_name}: {e}")
        return False

if __name__ == "__main__":
    packages = [
        "transformers",
        "torch",
        "modelscope",
        "sentencepiece",
        "colorama",
        "tqdm",
        "numpy",
        "flask",
        "werkzeug",
        "flask_wtf",
        "wtforms",
        "dotenv"
    ]
    
    success = True
    for package in packages:
        if not test_import(package):
            success = False
    
    if success:
        print("\nAll packages imported successfully!")
    else:
        print("\nSome packages failed to import. Please check the error messages above.")
        sys.exit(1)