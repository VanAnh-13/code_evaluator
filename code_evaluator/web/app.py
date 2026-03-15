"""
Flask application factory for Code Evaluator web app
"""

import os
import sys
import logging
import tempfile

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check if Flask is installed
try:
    from flask import Flask, redirect, url_for
    from flask_wtf.csrf import CSRFProtect, CSRFError
except ImportError:
    print("[ERROR] Flask or Flask-WTF is not installed.")
    print("\nPlease install the required dependencies:")
    print("    pip install flask flask-wtf")
    sys.exit(1)

from code_evaluator.utils.file_utils import ALLOWED_EXTENSIONS


def create_app(config=None):
    """
    Application factory for creating Flask app

    Args:
        config: Optional configuration dictionary

    Returns:
        Flask application instance
    """
    web_dir = os.path.dirname(__file__)

    app = Flask(
        __name__,
        template_folder=os.path.join(web_dir, 'templates'),
        static_folder=os.path.join(web_dir, 'static'),
        static_url_path='/static'
    )

    # Configure secret key
    _secret_key = os.environ.get('SECRET_KEY')
    if not _secret_key:
        _secret_key = os.urandom(24)
        logging.warning('SECRET_KEY not set. Using auto-generated random key.')
    app.config['SECRET_KEY'] = _secret_key

    # Configure upload settings
    app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'code_analyzer_uploads')
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload size
    app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

    # Apply additional config
    if config:
        app.config.update(config)

    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Enable CSRF protection
    csrf = CSRFProtect(app)

    # Register blueprints
    from code_evaluator.web.routes import main
    app.register_blueprint(main)

    # Exempt API endpoints from CSRF (they use JSON, not form posts)
    from code_evaluator.web.routes import api_analyze
    csrf.exempt(api_analyze)

    # Register agent blueprint and exempt from CSRF
    from code_evaluator.web.agent_routes import agent_bp
    app.register_blueprint(agent_bp)
    csrf.exempt(agent_bp)

    # Register error handlers
    register_error_handlers(app)
    
    return app


def register_error_handlers(app):
    """
    Register error handlers for the application
    
    Args:
        app: Flask application instance
    """
    @app.errorhandler(413)
    def too_large(e):
        """Handle file too large error"""
        from flask import flash
        flash('File too large. Maximum size is 5MB.', 'error')
        return redirect(url_for('main.index'))

    @app.errorhandler(CSRFError)
    def handle_csrf_error(e):
        """Handle CSRF validation errors with status 400"""
        return 'CSRF token missing or invalid.', 400

    @app.errorhandler(500)
    def server_error(e):
        """Handle server errors"""
        from flask import flash
        flash('An error occurred while processing your request. Please try again.', 'error')
        return redirect(url_for('main.index'))


# Legacy support: Direct app instance for backward compatibility
app = None


def get_app():
    """
    Get or create the Flask app instance
    
    Returns:
        Flask application instance
    """
    global app
    if app is None:
        app = create_app()
    return app


# For backwards compatibility with existing run_web.py
if __name__ == '__main__':
    application = create_app()
    try:
        port = int(os.environ.get('PORT', 5000))
    except ValueError:
        logging.warning(f"Invalid PORT value '{os.environ.get('PORT')}'. Using default port 5000.")
        port = 5000
    application.run(debug=True, host='0.0.0.0', port=port)
