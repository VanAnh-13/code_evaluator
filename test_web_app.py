#!/usr/bin/env python3
"""
Test script to verify the web application can start with the Ollama fixes
"""

import sys
import os
import tempfile

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_web_app_initialization():
    """Test that the web app can be initialized without errors"""
    try:
        # Import the app
        from web_app.app import app, analyzer
        
        print("[SUCCESS] Web app imported successfully")
        
        # Check that analyzer is properly initialized
        if analyzer is not None:
            print(f"[SUCCESS] Analyzer initialized with model: {analyzer.model_name}")
            
            # Test that the analyzer can perform basic analysis
            test_code = 'print("Hello, World!")'
            result = analyzer.analyze_code(test_code, "python")
            
            if "language" in result:
                print(f"[SUCCESS] Analyzer can perform basic analysis (language: {result['language']})")
                return True
            else:
                print("[WARNING] Analyzer analysis returned unexpected result")
                return False
        else:
            print("[FAILED] Analyzer is None")
            return False
            
    except Exception as e:
        print(f"[FAILED] Failed to initialize web app: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_flask_routes():
    """Test that Flask routes are properly defined"""
    try:
        from web_app.app import app
        
        # Get all routes
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append(rule.rule)
        
        expected_routes = ['/', '/upload', '/analyze/<file_id>', '/history', '/clear_history']
        
        missing_routes = []
        for route in expected_routes:
            # Check if any route matches the pattern
            route_found = False
            for app_route in routes:
                if route.replace('<file_id>', '<string:file_id>') == app_route or route == app_route:
                    route_found = True
                    break
            
            if not route_found:
                missing_routes.append(route)
        
        if not missing_routes:
            print(f"[SUCCESS] All expected routes are defined: {expected_routes}")
            return True
        else:
            print(f"[FAILED] Missing routes: {missing_routes}")
            print(f"[INFO] Available routes: {routes}")
            return False
            
    except Exception as e:
        print(f"[FAILED] Failed to check Flask routes: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Web Application with Ollama fixes")
    print("=" * 60)
    
    tests = [
        ("Web App Initialization", test_web_app_initialization),
        ("Flask Routes Test", test_flask_routes)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"[FAILED] {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("[SUCCESS] All web app tests passed! The application is ready to run.")
        print("[INFO] You can now start the web server using: python run_web.py")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
