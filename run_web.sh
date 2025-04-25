#!/bin/bash

echo "Starting C++ Code Analyzer Web Application..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH."
    echo "Please install Python from https://www.python.org/downloads/"
    echo
    read -p "Press Enter to continue..."
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
python3 -c "import flask" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Flask not found. Installing dependencies..."
    python3 install_dependencies.py
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies."
        read -p "Press Enter to continue..."
        exit 1
    fi
fi

# Run the web application
echo "Starting web server..."
python3 run_web.py

read -p "Press Enter to continue..."