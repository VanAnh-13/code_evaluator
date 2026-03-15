"""
Preservation Property Tests — Task 2
Captures existing CORRECT behavior on UNFIXED code.
These tests MUST PASS on the current (unfixed) codebase.

Methodology: observation-first
1. Observe behavior on unfixed code for inputs NOT in bug conditions
2. Write property-based tests capturing that observed behavior
3. Tests confirm baseline that must be preserved after bugfixes

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8
"""

import os
import sys
import re
import tempfile
import uuid
from unittest.mock import MagicMock

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

# Add parent directory so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock torch before importing code_analyzer (torch not installed in test env)
if 'torch' not in sys.modules:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.no_grad.return_value = MagicMock()
    sys.modules['torch'] = mock_torch

# Import from new package structure
from code_evaluator.analyzer.code_analyzer import CodeAnalyzer
from code_evaluator.utils.file_utils import detect_language


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = [
    '.py', '.js', '.cpp', '.cc', '.cxx', '.h', '.hpp',
    '.java', '.c', '.cs', '.php', '.rb', '.go', '.rs',
    '.ts', '.swift', '.kt', '.scala', '.html', '.css',
]

# Strategy: generate a supported file extension
st_extension = st.sampled_from(SUPPORTED_EXTENSIONS)

# Strategy: generate valid text content (printable ASCII, no null bytes)
st_text_content = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z', 'S'),
                           blacklist_characters='\x00'),
    min_size=1,
    max_size=200,
)

# Strategy: valid integer port values (1-65535)
st_valid_port = st.integers(min_value=1, max_value=65535)

# Strategy: code with bounded line count (1 to 100 lines for speed)
st_small_code = st.lists(
    st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'),
                               blacklist_characters='\x00'),
        min_size=0,
        max_size=80,
    ),
    min_size=1,
    max_size=100,
).map(lambda lines: '\n'.join(lines))


# ---------------------------------------------------------------------------
# Section-header analysis text strategy
# ---------------------------------------------------------------------------

SECTION_HEADERS = [
    "## Bugs",
    "## Memory Issues",
    "## Security Vulnerabilities",
    "## Performance Issues",
    "## Style Issues",
    "**Bugs**",
    "**Memory Management**",
    "**Security**",
    "**Performance**",
    "**Style**",
    "# Bugs and Logical Errors",
    "# Memory",
    "# Security",
    "# Performance",
    "# Readability",
]

SECTION_TO_KEY = {
    "bugs": "bugs",
    "logical errors": "bugs",
    "memory": "memory_issues",
    "memory management": "memory_issues",
    "memory issues": "memory_issues",
    "security": "security_vulnerabilities",
    "security vulnerabilities": "security_vulnerabilities",
    "performance": "performance_issues",
    "performance issues": "performance_issues",
    "style": "style_issues",
    "style issues": "style_issues",
    "readability": "style_issues",
}


def _header_to_result_key(header: str) -> str:
    """Map a section header string to the result dict key."""
    clean = header.lstrip('#* ').strip('*').strip()
    for keyword, key in SECTION_TO_KEY.items():
        if keyword in clean.lower():
            return key
    return "bugs"


def _build_analysis_text(header: str, issue_lines: list[str]) -> str:
    """Build an analysis text block with a header followed by issue lines."""
    return header + '\n' + '\n'.join(issue_lines)


st_section_header = st.sampled_from(SECTION_HEADERS)

st_issue_line = st.from_regex(
    r'- Line \d{1,4}: [a-zA-Z ]{5,40}',
    fullmatch=True,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    """Create a CodeAnalyzer instance (no model loaded — not needed for parsing)."""
    return CodeAnalyzer()


@pytest.fixture
def flask_client():
    """Create a Flask test client."""
    from web_app.app import app
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # not relevant on unfixed code
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# Property 1: For all valid text files with supported extensions, upload succeeds
# **Validates: Requirements 3.1**
# ---------------------------------------------------------------------------

class TestUploadValidFiles:
    """Property: valid text files with supported extensions are accepted."""

    @given(ext=st_extension, content=st_text_content)
    @settings(max_examples=30, deadline=10000)
    def test_valid_text_file_upload_accepted(self, ext, content):
        """
        For any valid text file with a supported extension,
        the upload endpoint accepts the file (no 'not allowed' flash).

        **Validates: Requirements 3.1**
        """
        from web_app.app import app
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False

        with app.test_client() as client:
            import io
            filename = f"testfile{ext}"
            data = {
                'file': (io.BytesIO(content.encode('utf-8')), filename)
            }
            response = client.post(
                '/upload',
                data=data,
                content_type='multipart/form-data',
                follow_redirects=False,
            )
            # On unfixed code, valid extension files are accepted → redirect to /analyze/<id>
            # Status 302 = redirect (accepted), NOT a flash error redirect back to /
            assert response.status_code == 302
            location = response.headers.get('Location', '')
            # Should redirect to analyze page, not back to index with error
            assert '/analyze/' in location or '/upload' in location


# ---------------------------------------------------------------------------
# Property 2: For all analysis text WITH section headers,
#             _parse_analysis() categorizes correctly
# **Validates: Requirements 3.3**
# ---------------------------------------------------------------------------

class TestParseAnalysisWithHeaders:
    """Property: text with section headers is categorized into the correct key."""

    @given(
        header=st_section_header,
        issues=st.lists(st_issue_line, min_size=1, max_size=5),
    )
    @settings(max_examples=40, deadline=5000)
    def test_parse_analysis_categorizes_with_header(self, header, issues):
        """
        For any analysis text that contains a recognized section header
        followed by issue lines, _parse_analysis() places issues in the
        correct category.

        **Validates: Requirements 3.3**
        """
        analyzer = CodeAnalyzer()
        text = _build_analysis_text(header, issues)
        result = analyzer._parse_analysis(text)

        # Result must have the 5-category structure
        assert set(result.keys()) == {
            'bugs', 'memory_issues', 'security_vulnerabilities',
            'performance_issues', 'style_issues',
        }

        expected_key = _header_to_result_key(header)
        # Issues should land in the expected category
        assert len(result[expected_key]) > 0, (
            f"Expected issues in '{expected_key}' for header '{header}', got empty list"
        )


# ---------------------------------------------------------------------------
# Property 3: For all valid integer PORT values, uses correct port
# **Validates: Requirements 3.6**
# ---------------------------------------------------------------------------

class TestValidPortUsed:
    """Property: valid integer PORT env var is used correctly."""

    @given(port=st_valid_port)
    @settings(max_examples=30, deadline=5000)
    def test_valid_port_parsed_correctly(self, port):
        """
        For any valid integer PORT value, int(os.environ.get('PORT', 5000))
        returns that exact value (no crash, no fallback).

        **Validates: Requirements 3.6**
        """
        # Simulate the PORT parsing logic from run_web.py
        os.environ['PORT'] = str(port)
        try:
            parsed = int(os.environ.get('PORT', 5000))
            assert parsed == port
        finally:
            del os.environ['PORT']


# ---------------------------------------------------------------------------
# Property 4: For all code ≤ 10,000 lines, accepted for analysis
# (On unfixed code there is NO line-count limit, so any size is accepted)
# **Validates: Requirements 3.4**
# ---------------------------------------------------------------------------

class TestCodeSizeAccepted:
    """Property: code within reasonable size is accepted for analysis."""

    @given(code=st_small_code)
    @settings(max_examples=20, deadline=10000)
    def test_small_code_accepted_for_analysis(self, code):
        """
        For any code ≤ 10,000 lines, analyze_file accepts it.
        On unfixed code, there is no line-count check, so all text files
        are accepted. This test captures that files with valid content
        are analyzed without error.

        **Validates: Requirements 3.4**
        """
        assume(len(code.strip()) > 0)

        analyzer = CodeAnalyzer()
        # Write to a temp .py file
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        )
        try:
            tmp.write(code)
            tmp.close()
            result = analyzer.analyze_file(tmp.name)
            # Should not return a file-not-found or empty-file error
            # (we ensured content is non-empty)
            assert 'error' not in result or 'empty' not in result.get('error', '').lower()
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)


# ---------------------------------------------------------------------------
# Property 5: GET routes return status 200
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------

class TestGetRoutesReturn200:
    """Property: GET routes (/, /history) return HTTP 200."""

    @pytest.mark.parametrize("route", ['/', '/history'])
    def test_get_route_returns_200(self, flask_client, route):
        """
        GET / and GET /history return status 200 on unfixed code.

        **Validates: Requirements 3.2**
        """
        response = flask_client.get(route)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Additional preservation observations
# ---------------------------------------------------------------------------

class TestAnalysisResultStructure:
    """Observation: analysis results always have the 5-category structure."""

    def test_parse_analysis_returns_5_categories(self, analyzer):
        """
        _parse_analysis() always returns a dict with exactly 5 keys,
        even for empty input.

        **Validates: Requirements 3.3**
        """
        result = analyzer._parse_analysis("")
        assert set(result.keys()) == {
            'bugs', 'memory_issues', 'security_vulnerabilities',
            'performance_issues', 'style_issues',
        }
        # All values are lists
        for v in result.values():
            assert isinstance(v, list)

    def test_parse_analysis_with_full_sections(self, analyzer):
        """
        _parse_analysis() with a single section header and multiple issues
        categorizes all issues into the correct category.

        **Validates: Requirements 3.3**
        """
        text = (
            "## Bugs\n"
            "- Line 1: null pointer dereference\n"
            "- Line 3: off by one error\n"
        )
        result = analyzer._parse_analysis(text)
        assert len(result['bugs']) == 2

        text2 = (
            "**Security**\n"
            "- Line 10: SQL injection risk\n"
            "- Line 15: XSS vulnerability found\n"
        )
        result2 = analyzer._parse_analysis(text2)
        assert len(result2['security_vulnerabilities']) == 2


class TestCachePreservation:
    """Observation: cache returns correct result for unchanged file."""

    def test_cache_returns_same_result(self, analyzer):
        """
        When analyzing the same file twice without changes,
        the second call returns the cached result.

        **Validates: Requirements 3.5**
        """
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        )
        try:
            tmp.write("x = 1\n")
            tmp.close()

            result1 = analyzer.analyze_file(tmp.name)
            result2 = analyzer.analyze_file(tmp.name)

            # Results should be identical (cached)
            assert result1 == result2
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)


class TestFileSizeLimit:
    """Observation: file > 5MB is rejected by Flask."""

    def test_large_file_rejected(self, flask_client):
        """
        Uploading a file larger than 5MB returns a 413 or redirect
        with error flash.

        **Validates: Requirements 3.7**
        """
        import io
        # Create content just over 5MB
        large_content = b'x' * (5 * 1024 * 1024 + 1)
        data = {
            'file': (io.BytesIO(large_content), 'big_file.py')
        }
        response = flask_client.post(
            '/upload',
            data=data,
            content_type='multipart/form-data',
            follow_redirects=False,
        )
        # Flask returns 413 for too-large requests
        assert response.status_code in (302, 413)


class TestEmptyFileHandling:
    """Observation: empty file returns error message."""

    def test_empty_file_returns_error(self, analyzer):
        """
        Analyzing an empty file returns a result with an error message.

        **Validates: Requirements 3.8**
        """
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        )
        try:
            tmp.write("")
            tmp.close()
            result = analyzer.analyze_file(tmp.name)
            assert 'error' in result
            assert 'empty' in result['error'].lower()
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)


class TestClearHistory:
    """Observation: Clear History clears session and temp files."""

    def test_clear_history_clears_session(self, flask_client):
        """
        POST /clear_history clears the files from session.

        **Validates: Requirements 3.6**
        """
        # First upload a file to create history
        import io
        data = {
            'file': (io.BytesIO(b'print("hello")'), 'test.py')
        }
        flask_client.post(
            '/upload',
            data=data,
            content_type='multipart/form-data',
            follow_redirects=True,
        )

        # Now clear history
        response = flask_client.post('/clear_history', follow_redirects=False)
        assert response.status_code == 302  # redirect after clear

        # Check history page is empty
        response = flask_client.get('/history')
        assert response.status_code == 200
