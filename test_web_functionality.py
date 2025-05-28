#!/usr/bin/env python3
"""
Test script to verify the web application's functionality with a sample C++ file.
"""

import requests
import os
import time

def test_web_app():
    """Test the web application's file upload and analysis functionality."""
    
    print("[INFO] Testing web application functionality...")
    
    # Wait for the web app to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:5000')
            if response.status_code == 200:
                print("✅ Web application is running!")
                break
        except requests.exceptions.ConnectionError:
            print(f"[INFO] Waiting for web app to start... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("❌ Web application failed to start")
        return False
    
    # Test file upload and analysis
    test_file_path = 'test_sample.cpp'
    if not os.path.exists(test_file_path):
        print(f"❌ Test file {test_file_path} not found")
        return False
    
    print("[INFO] Testing file upload and analysis...")
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {'file': (test_file_path, f, 'text/plain')}
            response = requests.post('http://localhost:5000/analyze', files=files)
        
        if response.status_code == 200:
            print("✅ File upload and analysis successful!")
            # Check if the response contains analysis results
            if 'analysis' in response.text.lower() or 'result' in response.text.lower():
                print("✅ Analysis results found in response")
            return True
        else:
            print(f"❌ Analysis failed with status code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Error during file upload: {e}")
        return False

if __name__ == "__main__":
    success = test_web_app()
    if success:
        print("🎉 All web application tests passed!")
    else:
        print("💔 Web application tests failed!")
