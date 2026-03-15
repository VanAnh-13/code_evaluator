"""Integration tests for web API endpoints"""

import pytest
import json
from unittest.mock import patch, MagicMock
from code_evaluator.web.app import create_app


@pytest.fixture
def client():
    """Create test client"""
    app = create_app(config={'TESTING': True, 'WTF_CSRF_ENABLED': False})
    with app.test_client() as client:
        yield client


class TestWebAPIIntegration:
    """Test web API integration"""

    @pytest.mark.integration
    def test_index_page_loads(self, client):
        """Test that index page loads successfully"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Code Evaluator' in response.data or b'code' in response.data.lower()

    @pytest.mark.integration
    def test_history_page_loads(self, client):
        """Test that history page loads"""
        response = client.get('/history')
        assert response.status_code == 200

    @pytest.mark.integration
    @patch('code_evaluator.web.routes.get_analyzer')
    def test_api_analyze_endpoint(self, mock_get_analyzer, client):
        """Test /api/analyze endpoint"""
        # Setup mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_code.return_value = {
            "summary": "Test analysis",
            "overall_score": 85,
            "issues": []
        }
        mock_get_analyzer.return_value = mock_analyzer

        # Send request
        response = client.post('/api/analyze',
                              json={
                                  'code': 'def test(): pass',
                                  'language': 'python'
                              },
                              content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'overall_score' in data or 'summary' in data

    @pytest.mark.integration
    def test_api_analyze_missing_code(self, client):
        """Test API with missing code field"""
        response = client.post('/api/analyze',
                              json={'language': 'python'},
                              content_type='application/json')

        # Should return error
        assert response.status_code in [400, 422]

    @pytest.mark.integration
    def test_api_analyze_invalid_json(self, client):
        """Test API with invalid JSON"""
        response = client.post('/api/analyze',
                              data='invalid json',
                              content_type='application/json')

        assert response.status_code in [400, 415]
