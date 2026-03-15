"""
Flask web application for Code Analyzer
This application provides a web interface for analyzing code in multiple languages using the Qwen model

NOTE: This is a backward-compatible wrapper. The main code is now in code_evaluator/web/
"""

import os
import sys

# Add parent directory to path to import new package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from new package structure
from code_evaluator.web import create_app
from code_evaluator.analyzer.code_analyzer import CodeAnalyzer
from code_evaluator.report.generator import generate_report

# Create app instance for backward compatibility
app = create_app()

# Export commonly used items
__all__ = ['app', 'CodeAnalyzer', 'generate_report']

if __name__ == '__main__':
    import logging
    try:
        port = int(os.environ.get('PORT', 5000))
    except ValueError:
        logging.warning(f"Invalid PORT value '{os.environ.get('PORT')}'. Using default port 5000.")
        port = 5000
    app.run(debug=True, host='0.0.0.0', port=port)
