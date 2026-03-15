"""
Route handlers for web application
Handles HTTP routes, file upload, and API endpoints
"""

import os
import sys
import uuid
import json
import logging
from datetime import datetime

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, session, jsonify, current_app
)
from werkzeug.utils import secure_filename

# Add parent directories to path for imports
_package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _package_root not in sys.path:
    sys.path.insert(0, _package_root)

from code_evaluator.report import generate_report
from code_evaluator.web.validators import (
    allowed_file,
    validate_upload,
    validate_code_content,
    get_unique_filename,
)

# Create blueprint with static folder
main = Blueprint(
    'main',
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    static_url_path='/static'
)

# Global analyzer instance (initialized lazily)
_analyzer = None
_CodeAnalyzer = None


def get_analyzer():
    """Get or create the analyzer instance (uses APIConfig from env)"""
    global _analyzer, _CodeAnalyzer
    if _analyzer is None:
        if _CodeAnalyzer is None:
            from code_evaluator.analyzer.code_analyzer import CodeAnalyzer
            _CodeAnalyzer = CodeAnalyzer
        from code_evaluator.model.config import APIConfig
        config = APIConfig.from_env()
        _analyzer = _CodeAnalyzer(config=config)
    return _analyzer


def _get_provider_info() -> dict:
    """Get current provider info for template context"""
    try:
        from code_evaluator.model.config import APIConfig
        config = APIConfig.from_env()
        return {"provider": config.provider, "model": config.model}
    except Exception:
        return {"provider": "none", "model": ""}


# ------------------------------------------------------------------
# Page routes
# ------------------------------------------------------------------

@main.route('/')
def index():
    """Render the home page with code editor"""
    return render_template('index.html', provider=_get_provider_info())


@main.route('/history')
def history():
    """Display history of analyzed files"""
    files = session.get('files', [])
    return render_template('history.html', files=files, provider=_get_provider_info())


@main.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the history of analyzed files"""
    if 'files' in session:
        for file_info in session['files']:
            path = file_info.get('path', '')
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        session.pop('files', None)
    flash('History cleared', 'success')
    return redirect(url_for('main.index'))


# ------------------------------------------------------------------
# File upload route (traditional form post → server-rendered page)
# ------------------------------------------------------------------

@main.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('main.index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('main.index'))

    if not (file and allowed_file(file.filename)):
        flash('File type not allowed. Please upload a supported code file.', 'error')
        return redirect(url_for('main.index'))

    filename = secure_filename(file.filename)
    unique_filename = get_unique_filename(filename)
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)

    try:
        is_valid, error_msg = validate_upload(file_path)
        if not is_valid:
            os.remove(file_path)
            flash(error_msg, 'error')
            return redirect(url_for('main.index'))

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
        return redirect(url_for('main.analyze', file_id=file_id))

    except Exception:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise


@main.route('/analyze/<file_id>')
def analyze(file_id):
    """Analyze the uploaded file and display results"""
    if 'files' not in session:
        flash('No file found', 'error')
        return redirect(url_for('main.index'))

    file_info = next((f for f in session['files'] if f['id'] == file_id), None)
    if not file_info:
        flash('File not found', 'error')
        return redirect(url_for('main.index'))

    file_path = file_info['path']

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            code_content = f.read()
        is_valid, error_msg = validate_code_content(code_content)
        if not is_valid:
            flash(error_msg, 'error')
            return redirect(url_for('main.index'))
    except (IOError, OSError):
        pass

    analyzer = get_analyzer()
    if not analyzer.model_loaded:
        load_result = analyzer.load_model()
        if not load_result:
            flash('API client is not available. Analysis limited to syntax checking.', 'warning')

    try:
        results = analyzer.analyze_file(file_path)
        report = generate_report(results)

        file_info['results'] = results
        file_info['score'] = results.get('overall_score', 0)
        file_info['language'] = results.get('language', 'unknown')
        file_info['issue_count'] = sum(
            len(results.get(k, []))
            for k in ['syntax_errors', 'bugs', 'memory_issues',
                      'security_vulnerabilities', 'performance_issues', 'style_issues']
        )
        session.modified = True

        return render_template(
            'analysis.html',
            file_info=file_info,
            results=results,
            report=report,
            provider=_get_provider_info()
        )
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


# ------------------------------------------------------------------
# API endpoint (AJAX from editor page)
# ------------------------------------------------------------------

@main.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    JSON API for inline code analysis.
    Expects: { "code": "...", "language": "auto|cpp|python|..." }
    Returns: JSON analysis results
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    code = data.get('code', '').strip()
    language = data.get('language', 'auto').strip()

    if not code:
        return jsonify({"error": "No code provided"}), 400

    is_valid, error_msg = validate_code_content(code)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    analyzer = get_analyzer()
    if not analyzer.model_loaded:
        if not analyzer.load_model():
            return jsonify({"error": "API client failed to initialize. Check your API_KEY and API_PROVIDER settings."}), 503

    try:
        results = analyzer.analyze_code(code, language)

        # Flatten issues into a single list for the frontend
        all_issues = []
        category_keys = {
            'syntax_errors': 'syntax',
            'bugs': 'bugs',
            'memory_issues': 'memory',
            'security_vulnerabilities': 'security',
            'performance_issues': 'performance',
            'style_issues': 'style',
        }
        for key, cat_name in category_keys.items():
            for issue in results.get(key, []):
                issue_copy = dict(issue)
                if 'category' not in issue_copy:
                    issue_copy['category'] = cat_name
                all_issues.append(issue_copy)

        return jsonify({
            "language": results.get('detected_language', results.get('language', language)),
            "summary": results.get('summary', ''),
            "overall_score": results.get('overall_score', 0),
            "issues": all_issues,
            "suggested_fixes": results.get('suggested_fixes', []),
        })

    except Exception as e:
        logging.error(f"API analyze error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
