"""
Bug Condition Exploration Tests for Code Analyzer
==================================================
These tests encode the EXPECTED (correct) behavior for 14 identified bugs.
They are designed to FAIL on the unfixed code, confirming the bugs exist.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
             1.9, 1.10, 1.11, 1.12, 1.13, 1.14**

Property 1: Fault Condition — Khám phá 14 lỗi trên code chưa sửa
"""

import os
import sys
import ast
import tempfile
import inspect
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock torch before importing code_analyzer (torch not installed in test env)
if 'torch' not in sys.modules:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.no_grad.return_value = MagicMock()
    sys.modules['torch'] = mock_torch


def _reimport_web_app():
    """Force reimport of web_app to pick up env/mock changes."""
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith('web_app') or mod_name.startswith('code_evaluator.web'):
            del sys.modules[mod_name]
    from code_evaluator.web import create_app
    return create_app()


# ============================================================================
# Nhóm A — Bảo mật (Security)
# ============================================================================

class TestGroupA_Security:
    """Nhóm A — Bảo mật: Tests for security-related bugs."""

    def test_secret_key_not_hardcoded(self):
        """
        Bug 1.1: SECRET_KEY should NOT be hardcoded 'dev_key_for_development_only'
        when SECRET_KEY env var is not set.

        **Validates: Requirements 1.1**
        """
        env = os.environ.copy()
        env.pop('SECRET_KEY', None)

        with patch.dict(os.environ, env, clear=True):
            app = _reimport_web_app()
            secret_key = app.config['SECRET_KEY']
            assert secret_key != 'dev_key_for_development_only', (
                f"SECRET_KEY is hardcoded to 'dev_key_for_development_only'. "
                f"Should be randomly generated when env var is not set."
            )

    def test_csrf_protection_on_post(self):
        """
        Bug 1.2: POST to /upload without CSRF token should return 400.

        **Validates: Requirements 1.2**
        """
        app = _reimport_web_app()
        app.config['TESTING'] = True

        with app.test_client() as client:
            response = client.post('/upload', data={},
                                   content_type='multipart/form-data')
            assert response.status_code == 400, (
                f"POST /upload without CSRF token returned {response.status_code}, "
                f"expected 400. CSRF protection is missing."
            )

    def test_binary_upload_rejected(self):
        """
        Bug 1.3: Upload .py file with binary content should be rejected.

        **Validates: Requirements 1.3**
        """
        app = _reimport_web_app()
        app.config['TESTING'] = True

        binary_content = b'\x00\x01\x02\x03\x89PNG\r\n\x1a\n' + b'\x00' * 100

        with app.test_client() as client:
            from io import BytesIO
            data = {'file': (BytesIO(binary_content), 'malicious.py')}
            response = client.post('/upload', data=data,
                                   content_type='multipart/form-data')
            if response.status_code == 302:
                location = response.headers.get('Location', '')
                assert 'analyze' not in location, (
                    f"Binary file with .py extension was accepted for analysis. "
                    f"Binary content should be rejected."
                )

    def test_path_traversal_rejected(self):
        """
        Bug 1.4: analyze_file with path traversal should be rejected.

        **Validates: Requirements 1.4**
        """
        from code_analyzer import CodeAnalyzer
        analyzer = CodeAnalyzer()

        result = analyzer.analyze_file("../../etc/passwd")

        assert 'error' in result, "analyze_file should return an error for path traversal"
        error_msg = result['error'].lower()
        assert any(term in error_msg for term in [
            'path traversal', 'not allowed', 'security', 'outside', 'forbidden'
        ]), (
            f"analyze_file('../../etc/passwd') returned: '{result['error']}'. "
            f"Should explicitly reject path traversal."
        )


# ============================================================================
# Nhóm B — Logic
# ============================================================================

class TestGroupB_Logic:
    """Nhóm B — Logic: Tests for logic-related bugs."""

    def test_parse_no_header_returns_results(self):
        """
        Bug 1.5: _parse_analysis with no section header should return results.

        **Validates: Requirements 1.5**
        """
        from code_analyzer import CodeAnalyzer
        analyzer = CodeAnalyzer()

        text = "- Line 5: bug found\n- Line 10: null pointer dereference"
        result = analyzer._parse_analysis(text)

        total_issues = sum(len(v) for v in result.values())
        assert total_issues > 0, (
            f"_parse_analysis returned no issues for text without section headers. "
            f"Result: {result}. Issues should be assigned to default 'bugs' section."
        )

    def test_load_model_failure_handled(self):
        """
        Bug 1.6: load_model() failure at web startup should be handled.

        **Validates: Requirements 1.6**
        """
        app_source_path = os.path.join(os.path.dirname(__file__),
                                       'web_app', 'app.py')
        with open(app_source_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        has_load_model_check = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_source = ast.dump(node.test)
                if 'load_model' in test_source or 'model_loaded' in test_source:
                    has_load_model_check = True
                    break

        load_model_result_checked = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Attribute):
                        if node.value.func.attr == 'load_model':
                            load_model_result_checked = True
                            break

        assert has_load_model_check or load_model_result_checked, (
            "web_app/app.py calls analyzer.load_model() but does not check "
            "the return value or handle failure. Expected: check result and "
            "log error / notify user if model fails to load."
        )

    def test_cache_collision_different_content(self):
        """
        Bug 1.7: Files with different content but same path+mtime should
        produce different analysis results (content-based cache keys).

        **Validates: Requirements 1.7**
        """
        from code_analyzer import CodeAnalyzer

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as f:
            temp_path = f.name
            f.write('x = 1  # version 1')

        try:
            analyzer = CodeAnalyzer()

            # Analyze the file with content "version 1"
            result1 = analyzer.analyze_file(temp_path)

            # Overwrite with different content, preserve mtime
            mtime = os.path.getmtime(temp_path)
            with open(temp_path, 'w') as f:
                f.write('y = 2  # version 2 completely different')
            os.utime(temp_path, (mtime, mtime))

            # Analyze again — should NOT return cached result from version 1
            result2 = analyzer.analyze_file(temp_path)

            # If cache uses content hash, result2 should reflect the new content
            # (i.e., not be the exact same cached object from result1)
            # The file_path is the same, but the analysis should be re-run
            # because content changed. We verify by checking they are not
            # the exact same object (cache miss means fresh analysis).
            assert result1 is not result2, (
                "Cache returned the same object for different file content. "
                "Cache key should include content hash, not just path+mtime."
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_port_invalid_no_crash(self):
        """
        Bug 1.8: PORT=abc should not crash, should fallback to 5000.

        **Validates: Requirements 1.8**
        """
        run_web_path = os.path.join(os.path.dirname(__file__), 'run_web.py')
        with open(run_web_path, 'r') as f:
            run_web_source = f.read()

        app_path = os.path.join(os.path.dirname(__file__), 'web_app', 'app.py')
        with open(app_path, 'r') as f:
            app_source = f.read()

        has_port_validation = False
        for source in [run_web_source, app_source]:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    try_source = ast.dump(node)
                    if 'PORT' in try_source and 'ValueError' in try_source:
                        has_port_validation = True
                        break

        assert has_port_validation, (
            "No try/except ValueError around int(PORT) in run_web.py or app.py. "
            "PORT=abc will cause ValueError crash. "
            "Expected: catch ValueError and fallback to port 5000."
        )


# ============================================================================
# Nhóm C — Tài nguyên (Resources)
# ============================================================================

class TestGroupC_Resources:
    """Nhóm C — Tài nguyên: Tests for resource management bugs."""

    def test_temp_file_cleanup_on_exception(self):
        """
        Bug 1.9: Temp file should be cleaned up when exception occurs.

        **Validates: Requirements 1.9**
        """
        app = _reimport_web_app()
        app.config['TESTING'] = True

        with app.test_client() as client:
            from io import BytesIO
            data = {'file': (BytesIO(b'print("hello")'), 'test_cleanup.py')}

            with patch('code_analyzer.CodeAnalyzer.analyze_file',
                       side_effect=RuntimeError("Analysis failed")):
                response = client.post('/upload', data=data,
                                       content_type='multipart/form-data')
                if response.status_code == 302:
                    location = response.headers.get('Location', '')
                    if 'analyze' in location:
                        try:
                            client.get(location)
                        except Exception:
                            pass

            upload_folder = app.config['UPLOAD_FOLDER']
            remaining = os.listdir(upload_folder) if os.path.exists(upload_folder) else []
            test_files = [f for f in remaining if 'test_cleanup' in f.lower()]

            assert len(test_files) == 0, (
                f"Temp files remain after exception: {test_files}. "
                f"Expected: temp files cleaned up in finally block."
            )

    def test_auto_delete_upload_after_analysis(self):
        """
        Bug 1.10: Uploaded file should be auto-deleted after analysis.

        **Validates: Requirements 1.10**
        """
        app = _reimport_web_app()
        app.config['TESTING'] = True

        mock_results = {
            "language": "python", "syntax_errors": [],
            "bugs": [], "memory_issues": [],
            "security_vulnerabilities": [], "performance_issues": [],
            "style_issues": [], "file_path": "test.py"
        }

        with app.test_client() as client:
            from io import BytesIO
            data = {'file': (BytesIO(b'x = 1'), 'test_auto_delete.py')}

            with patch('code_analyzer.CodeAnalyzer.analyze_file',
                       return_value=mock_results):
                response = client.post('/upload', data=data,
                                       content_type='multipart/form-data')
                if response.status_code == 302:
                    location = response.headers.get('Location', '')
                    if 'analyze' in location:
                        client.get(location)

            upload_folder = app.config['UPLOAD_FOLDER']
            remaining = os.listdir(upload_folder) if os.path.exists(upload_folder) else []
            auto_files = [f for f in remaining if 'test_auto_delete' in f]

            assert len(auto_files) == 0, (
                f"Uploaded file not auto-deleted after analysis: {auto_files}. "
                f"Expected: file removed after analysis completes."
            )

    def test_model_timeout(self):
        """
        Bug 1.11: model.generate() should timeout after 120 seconds.

        **Validates: Requirements 1.11**
        """
        from code_analyzer import CodeAnalyzer
        source = inspect.getsource(CodeAnalyzer.analyze_code)

        has_timeout = any(term in source.lower() for term in [
            'timeout', 'timer', 'signal.alarm', 'threading.timer',
            'time_limit', 'max_time', 'deadline'
        ])

        assert has_timeout, (
            "analyze_code() has no timeout mechanism for model.generate(). "
            "Expected: timeout of 120 seconds."
        )

    def test_lazy_loading_model(self):
        """
        Bug 1.12: Model should NOT be loaded at web app startup.

        **Validates: Requirements 1.12**
        """
        app_path = os.path.join(os.path.dirname(__file__), 'web_app', 'app.py')
        with open(app_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        module_level_load_model = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute):
                    if call.func.attr == 'load_model':
                        module_level_load_model = True
                        break

        assert not module_level_load_model, (
            "analyzer.load_model() called at module level in web_app/app.py. "
            "Expected: lazy loading on first analysis request."
        )


# ============================================================================
# Nhóm D — Chất lượng (Quality)
# ============================================================================

class TestGroupD_Quality:
    """Nhóm D — Chất lượng: Tests for code quality bugs."""

    def test_unused_import_jsonify(self):
        """
        Bug 1.13: web_app/app.py should not import jsonify (unused).

        **Validates: Requirements 1.13**
        """
        app_path = os.path.join(os.path.dirname(__file__), 'web_app', 'app.py')
        with open(app_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        imports_jsonify = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'flask':
                for alias in node.names:
                    if alias.name == 'jsonify':
                        imports_jsonify = True
                        break

        jsonify_called = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'jsonify':
                    jsonify_called = True
                    break

        assert not imports_jsonify or jsonify_called, (
            "web_app/app.py imports 'jsonify' from flask but never uses it. "
            "Expected: only import modules that are actually used."
        )

    def test_code_size_limit(self):
        """
        Bug 1.14: Code with 50,000 lines should be rejected.

        **Validates: Requirements 1.14**
        """
        app = _reimport_web_app()
        app.config['TESTING'] = True
        app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

        large_code = '\n'.join([f'x = {i}' for i in range(50000)])

        mock_results = {
            "language": "python", "syntax_errors": [],
            "bugs": [], "memory_issues": [],
            "security_vulnerabilities": [], "performance_issues": [],
            "style_issues": [], "file_path": "test.py"
        }

        with app.test_client() as client:
            from io import BytesIO
            data = {'file': (BytesIO(large_code.encode('utf-8')), 'large.py')}

            with patch('code_analyzer.CodeAnalyzer.analyze_file',
                       return_value=mock_results) as mock_analyze:
                response = client.post('/upload', data=data,
                                       content_type='multipart/form-data')
                if response.status_code == 302:
                    location = response.headers.get('Location', '')
                    if 'analyze' in location:
                        client.get(location)
                        if mock_analyze.called:
                            pytest.fail(
                                "50,000 line code was accepted for analysis. "
                                "Expected: code > 10,000 lines rejected."
                            )
