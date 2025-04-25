"""
Run script for the C++ Code Analyzer web application
"""

import os
import sys
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['flask', 'werkzeug']
    missing_packages = []

    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)

    return missing_packages

if __name__ == "__main__":
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

    try:
        # Import Flask app after dependency check
        from web_app.app import app

        # Get port from environment variable or use default
        port = int(os.environ.get('PORT', 5000))

        # Print startup message
        print(f"[INFO] Starting C++ Code Analyzer web server on http://localhost:{port}")
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
