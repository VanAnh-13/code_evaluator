"""
Run script for the Code Analyzer web application
This script starts the web server for analyzing code in multiple programming languages
"""

import os
import sys
import importlib.util
import platform
import tempfile

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['flask', 'werkzeug']
    missing_packages = []

    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)

    return missing_packages

def check_environment():
    """Check if the environment is properly set up"""
    issues = []

    # Check Python version
    python_version = platform.python_version()
    if int(python_version.split('.')[0]) < 3 or (int(python_version.split('.')[0]) == 3 and int(python_version.split('.')[1]) < 6):
        issues.append(f"Python version {python_version} is too old. Python 3.6 or newer is required.")

    # Check if temp directory is writable
    temp_dir = os.path.join(tempfile.gettempdir(), 'code_analyzer_uploads')
    try:
        os.makedirs(temp_dir, exist_ok=True)
        test_file = os.path.join(temp_dir, 'test_write.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        issues.append(f"Cannot write to temporary directory: {str(e)}")

    return issues

if __name__ == "__main__":
    print("[INFO] Starting Code Analyzer web server...")

    # Check if the required directories exist
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'web_app')):
        print("[ERROR] Web application directory not found. Make sure you're running this script from the code_evaluator directory.")
        sys.exit(1)

    # Check for required dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print("[ERROR] Missing required dependencies: " + ", ".join(missing_packages))
        print("\nPlease install the required dependencies using one of the following methods:")
        print("\nMethod 1 (Recommended):")
        print("    python install_dependencies.py")
        print("\nMethod 2:")
        print("    pip install -r requirements.txt")
        sys.exit(1)

    # Check environment
    env_issues = check_environment()
    if env_issues:
        print("[ERROR] Environment issues detected:")
        for issue in env_issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before running the web server.")
        sys.exit(1)

    print("[INFO] Environment check passed. Starting web server...")

    try:
        # Import Flask app after dependency check
        from web_app.app import app

        # Get port from environment variable or use default
        port = int(os.environ.get('PORT', 5000))

        # Print startup message
        print(f"[INFO] Starting Code Analyzer web server on http://localhost:{port}")
        print("[INFO] You can analyze code in multiple languages including C++, Python, JavaScript, Java, and more")
        print("[INFO] Press Ctrl+C to stop the server")

        # Run the Flask application
        app.run(debug=False, host='0.0.0.0', port=port)
    except ModuleNotFoundError as e:
        print(f"[ERROR] {e}")
        print("\nPlease install the required dependencies using one of the following methods:")
        print("\nMethod 1 (Recommended):")
        print("    python install_dependencies.py")
        print("\nMethod 2:")
        print("    pip install -r requirements.txt")
        sys.exit(1)
