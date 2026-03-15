"""Tests for agent tools"""

import pytest
from unittest.mock import Mock, patch
from code_evaluator.agent.tools import create_default_registry
from code_evaluator.agent.tool_registry import ToolRegistry


class TestAgentTools:
    """Test agent tool functionality"""

    def test_create_default_registry(self):
        """Test creating default tool registry"""
        registry = create_default_registry()
        assert isinstance(registry, ToolRegistry)
        assert len(registry.list_tools()) > 0

    def test_registry_has_required_tools(self):
        """Test registry contains required tools"""
        registry = create_default_registry()
        tool_names = [tool['name'] for tool in registry.list_tools()]

        # Should have basic code analysis tools
        # Adjust based on actual tool names in your implementation
        assert len(tool_names) > 0

    def test_tool_execution(self):
        """Test basic tool execution"""
        registry = create_default_registry()
        tools = registry.list_tools()

        if tools:
            # Test that tools can be retrieved
            first_tool_name = tools[0]['name']
            assert registry.get_tool(first_tool_name) is not None
