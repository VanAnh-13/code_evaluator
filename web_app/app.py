"""
Flask web application for C++ Code Analyzer
This application provides a web interface for analyzing C++ code using the Qwen model
"""

import os
import sys
import uuid
import tempfile
import importlib.util

# Check if Flask is installed
try:
    from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
    from werkzeug.utils import secure_filename
except ImportError:
    print("[ERROR] Flask or Werkzeug is not installed.")
    print("\nPlease install the required dependencies using one of the following methods:")
    print("\nMethod 1 (Recommended):")
    print("    python install_dependencies.py")
    print("\nMethod 2:")
    print("    pip install -r requirements.txt")
    sys.exit(1)

from datetime import datetime

# Add parent directory to path to import cpp_code_analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cpp_code_analyzer import CppCodeAnalyzer, generate_report

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_development_only')
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'cpp_analyzer_uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'.cpp', '.cc', '.cxx', '.h', '.hpp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the analyzer
analyzer = CppCodeAnalyzer()
analyzer.load_model()

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return os.path.splitext(filename)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_unique_filename(filename):
    """Generate a unique filename to prevent overwriting"""
    base, ext = os.path.splitext(filename)
    return f"{base}_{uuid.uuid4().hex}{ext}"

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Secure the filename and make it unique
        filename = secure_filename(file.filename)
        unique_filename = get_unique_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Save the file
        file.save(file_path)

        # Store file info in session
        if 'files' not in session:
            session['files'] = []

        file_id = str(uuid.uuid4())
        session['files'].append({
            'id': file_id,
            'original_name': filename,
            'path': file_path,
            'timestamp': datetime.now().isoformat()
        })
        session.modified = True

        # Redirect to analysis page
        return redirect(url_for('analyze', file_id=file_id))

    flash('File type not allowed. Please upload a C++ file (.cpp, .cc, .cxx, .h, .hpp)', 'error')
    return redirect(url_for('index'))

@app.route('/analyze/<file_id>')
def analyze(file_id):
    """Analyze the uploaded file and display results"""
    # Find the file in session
    if 'files' not in session:
        flash('No file found', 'error')
        return redirect(url_for('index'))

    file_info = next((f for f in session['files'] if f['id'] == file_id), None)
    if not file_info:
        flash('File not found', 'error')
        return redirect(url_for('index'))

    # Analyze the file
    results = analyzer.analyze_file(file_info['path'])

    # Generate report
    report = generate_report(results)

    # Store results in session
    file_info['results'] = results
    session.modified = True

    return render_template('analysis.html', 
                          file_info=file_info, 
                          results=results, 
                          report=report)

@app.route('/history')
def history():
    """Display history of analyzed files"""
    files = session.get('files', [])
    return render_template('history.html', files=files)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the history of analyzed files"""
    if 'files' in session:
        # Delete temporary files
        for file_info in session['files']:
            if os.path.exists(file_info['path']):
                try:
                    os.remove(file_info['path'])
                except:
                    pass

        # Clear session
        session.pop('files', None)

    flash('History cleared', 'success')
    return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Maximum size is 5MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    flash('An error occurred while processing your request. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
