#!/usr/bin/env python3

import requests
import json
import os

def test_web_cpp_syntax_checking():
    """Test C++ syntax checking through the web application"""
    
    # Read the example C++ file with syntax errors
    cpp_file_path = os.path.join("examples", "example.cpp")
    
    if not os.path.exists(cpp_file_path):
        print(f"Error: {cpp_file_path} not found")
        return
        
    with open(cpp_file_path, 'r') as f:
        cpp_code = f.read()
    
    print("Testing C++ syntax checking through web API...")
    print(f"Code to test:\n{cpp_code}\n")
    
    # Send POST request to the web application
    url = "http://localhost:5000/analyze"
    data = {
        'code': cpp_code,
        'language': 'cpp'
    }
    
    try:
        response = requests.post(url, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("Analysis completed successfully!")
            print(f"Total issues found: {len(result.get('syntax_errors', [])) + len(result.get('bugs', [])) + len(result.get('memory_issues', [])) + len(result.get('security_vulnerabilities', [])) + len(result.get('performance_issues', [])) + len(result.get('style_issues', []))}")
            
            # Check syntax errors specifically
            syntax_errors = result.get('syntax_errors', [])
            print(f"\nSyntax errors found: {len(syntax_errors)}")
            
            for i, error in enumerate(syntax_errors, 1):
                print(f"{i}. Line {error.get('line', 'N/A')}: {error.get('description', 'No description')}")
                print(f"   Severity: {error.get('severity', 'N/A')}")
                print(f"   Recommendation: {error.get('recommendation', 'No recommendation')}")
                print()
            
            if len(syntax_errors) >= 2:
                print("✅ SUCCESS: C++ syntax checking is working correctly through the web interface!")
                print("✅ Detected expected syntax errors (missing semicolon and missing brace)")
            else:
                print(f"❌ ISSUE: Expected at least 2 syntax errors, but found {len(syntax_errors)}")
                
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Failed to connect to web application: {e}")
        print("Make sure the web application is running on http://localhost:5000")

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir("d:/Homeworks/python/cpp_code_analyzer/code_evaluator")
    test_web_cpp_syntax_checking()
