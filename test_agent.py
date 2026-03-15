"""
Tests for the AI Agent module.
Covers: ToolRegistry, AgentSession, SessionManager, AgentExecutor (with mock LLM),
        and agent API routes.
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

class TestToolRegistry(unittest.TestCase):
    """Test the ToolRegistry and its decorator registration."""

    def setUp(self):
        from code_evaluator.agent.tool_registry import ToolRegistry
        self.registry = ToolRegistry()

    def test_register_via_decorator(self):
        @self.registry.tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
        )
        def handler(msg):
            return f"echo: {msg}"

        self.assertIn("test_tool", [t.name for t in self.registry._tools.values()])

    def test_execute_returns_string(self):
        @self.registry.tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
        )
        def handler(a, b):
            return a + b  # returns int

        result = self.registry.execute("add", {"a": 3, "b": 4})
        self.assertIsInstance(result, str)  # must be stringified
        self.assertIn("7", result)

    def test_execute_unknown_tool(self):
        with self.assertRaises(ValueError):
            self.registry.execute("nonexistent", {})

    def test_get_tool_definitions(self):
        @self.registry.tool(
            name="hello",
            description="Greet",
            parameters={"type": "object", "properties": {}},
        )
        def handler():
            return "hi"

        defs = self.registry.get_tool_definitions()
        self.assertEqual(len(defs), 1)
        self.assertEqual(defs[0]["type"], "function")
        self.assertEqual(defs[0]["function"]["name"], "hello")

    def test_list_tools(self):
        @self.registry.tool(
            name="t1", description="Tool 1",
            parameters={"type": "object", "properties": {}},
        )
        def h1():
            return ""

        @self.registry.tool(
            name="t2", description="Tool 2",
            parameters={"type": "object", "properties": {}},
        )
        def h2():
            return ""

        names = self.registry.list_tools()
        self.assertEqual(set(names), {"t1", "t2"})


# ---------------------------------------------------------------------------
# AgentSession & SessionManager tests
# ---------------------------------------------------------------------------

class TestAgentSession(unittest.TestCase):
    """Test AgentSession state management."""

    def test_initial_state(self):
        from code_evaluator.agent.session import AgentSession, SessionStatus
        s = AgentSession(max_steps=5)
        self.assertEqual(s.status, SessionStatus.PENDING)
        self.assertTrue(s.can_continue)
        self.assertEqual(len(s.messages), 0)
        self.assertEqual(len(s.steps), 0)

    def test_state_transitions(self):
        from code_evaluator.agent.session import AgentSession, SessionStatus
        s = AgentSession()
        s.set_running()
        self.assertEqual(s.status, SessionStatus.RUNNING)
        s.set_completed({"score": 85})
        self.assertEqual(s.status, SessionStatus.COMPLETED)
        self.assertEqual(s.result["score"], 85)

    def test_can_continue_max_steps(self):
        from code_evaluator.agent.session import AgentSession, AgentStep, StepType
        s = AgentSession(max_steps=2)
        s.set_running()
        self.assertTrue(s.can_continue)
        s.add_step(AgentStep(step_number=1, type=StepType.THINKING))
        s.add_step(AgentStep(step_number=2, type=StepType.RESPONSE))
        self.assertFalse(s.can_continue)

    def test_subscriber_notification(self):
        from code_evaluator.agent.session import AgentSession, AgentStep, StepType
        s = AgentSession()
        received = []
        s.subscribe(lambda step: received.append(step))
        step = AgentStep(step_number=1, type=StepType.THINKING, content="test")
        s.add_step(step)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].content, "test")

    def test_to_dict(self):
        from code_evaluator.agent.session import AgentSession
        s = AgentSession()
        d = s.to_dict()
        self.assertIn("session_id", d)
        self.assertIn("status", d)
        self.assertIn("steps", d)


class TestSessionManager(unittest.TestCase):
    """Test SessionManager CRUD and cleanup."""

    def test_create_and_get(self):
        from code_evaluator.agent.session import SessionManager
        sm = SessionManager(ttl=60)
        s = sm.create(max_steps=10)
        self.assertIsNotNone(sm.get(s.session_id))

    def test_delete(self):
        from code_evaluator.agent.session import SessionManager
        sm = SessionManager()
        s = sm.create()
        self.assertTrue(sm.delete(s.session_id))
        self.assertIsNone(sm.get(s.session_id))
        self.assertFalse(sm.delete("nonexistent"))

    def test_cleanup_expired(self):
        from code_evaluator.agent.session import SessionManager
        sm = SessionManager(ttl=0)  # expire immediately
        sm.create()
        sm.create()
        time.sleep(0.05)
        cleaned = sm.cleanup_expired()
        self.assertEqual(cleaned, 2)
        self.assertEqual(sm.active_count, 0)

    def test_list_sessions(self):
        from code_evaluator.agent.session import SessionManager
        sm = SessionManager()
        sm.create()
        sm.create()
        listing = sm.list_sessions()
        self.assertEqual(len(listing), 2)


# ---------------------------------------------------------------------------
# AgentExecutor tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestAgentExecutor(unittest.TestCase):
    """Test AgentExecutor with a mocked model_loader."""

    def _make_executor(self, responses):
        """Create an executor whose model_loader returns canned responses."""
        from code_evaluator.agent.executor import AgentExecutor
        from code_evaluator.agent.tool_registry import ToolRegistry
        from code_evaluator.agent.session import SessionManager

        registry = ToolRegistry()

        @registry.tool(
            name="echo",
            description="Echo input",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        def echo_handler(text):
            return f"echoed: {text}"

        model_loader = MagicMock()
        model_loader.is_loaded = True
        model_loader.load.return_value = True

        # responses is a list; pop from front each call
        model_loader.analyze.side_effect = list(responses)

        sm = SessionManager()
        executor = AgentExecutor(
            model_loader=model_loader,
            tool_registry=registry,
            session_manager=sm,
            system_prompt="You are a test agent.",
        )
        return executor, sm

    def test_simple_text_response(self):
        """Agent receives a plain text response and completes."""
        executor, sm = self._make_executor(["Final answer: all good."])
        session = sm.create(max_steps=10)
        steps = list(executor.run(session, "Analyze this code", code="print('hi')"))

        # Expect: THINKING + RESPONSE
        types = [s.type.value for s in steps]
        self.assertIn("thinking", types)
        self.assertIn("response", types)
        self.assertEqual(session.status.value, "completed")

    def test_tool_call_then_response(self):
        """Agent calls a tool, gets result, then gives text response."""
        from code_evaluator.model.base_client import ChatResponse, ToolCall

        tool_response = ChatResponse(
            content="",
            tool_calls=[
                ToolCall(id="tc_1", name="echo", arguments={"text": "hello"})
            ],
            raw={"role": "assistant", "content": ""},
        )
        text_response = "Analysis complete."

        executor, sm = self._make_executor([tool_response, text_response])
        session = sm.create(max_steps=20)
        steps = list(executor.run(session, "Test"))

        types = [s.type.value for s in steps]
        self.assertIn("tool_call", types)
        self.assertIn("tool_result", types)
        self.assertIn("response", types)
        self.assertEqual(session.status.value, "completed")

        # Verify tool was actually called with right args
        tool_result_steps = [s for s in steps if s.type.value == "tool_result"]
        self.assertTrue(any("echoed: hello" in s.tool_result for s in tool_result_steps))

    def test_max_steps_limit(self):
        """Agent stops when max steps is reached."""
        from code_evaluator.model.base_client import ChatResponse, ToolCall

        # Return tool calls forever
        def infinite_tool_calls(*args, **kwargs):
            return ChatResponse(
                content="",
                tool_calls=[ToolCall(id="tc_x", name="echo", arguments={"text": "loop"})],
                raw={"role": "assistant", "content": ""},
            )

        executor, sm = self._make_executor([])
        executor.model_loader.analyze.side_effect = infinite_tool_calls
        session = sm.create(max_steps=6)
        steps = list(executor.run(session, "Test"))

        # Should have hit the limit
        self.assertEqual(session.status.value, "completed")
        error_steps = [s for s in steps if s.type.value == "error"]
        self.assertTrue(any("maximum step limit" in s.content.lower() for s in error_steps))

    def test_continue_conversation(self):
        """Continue an existing session with follow-up."""
        from code_evaluator.agent.session import SessionStatus

        executor, sm = self._make_executor(["First response", "Follow-up response"])
        session = sm.create(max_steps=20)

        # First turn
        list(executor.run(session, "Initial query"))
        self.assertEqual(session.status.value, "completed")

        # Continue
        steps = list(executor.continue_conversation(session, "Follow-up question"))
        response_steps = [s for s in steps if s.type.value == "response"]
        self.assertTrue(len(response_steps) > 0)

    def test_run_analysis_convenience(self):
        """run_analysis creates session and produces result."""
        executor, sm = self._make_executor([
            json.dumps({
                "language": "python",
                "summary": "Clean code",
                "overall_score": 90,
                "issues": [],
                "suggested_fixes": [],
            })
        ])
        session = executor.run_analysis(code="x = 1", language="python")
        self.assertEqual(session.status.value, "completed")
        self.assertIsNotNone(session.result)


# ---------------------------------------------------------------------------
# Default tool registry — smoke test
# ---------------------------------------------------------------------------

class TestDefaultRegistry(unittest.TestCase):
    """Smoke test: default registry should register all expected tools."""

    def test_default_tools_registered(self):
        from code_evaluator.agent.tools import create_default_registry
        registry = create_default_registry()
        names = set(registry.list_tools())
        expected = {
            "check_syntax", "analyze_patterns", "suggest_fixes",
            "apply_fix", "verify_fix", "read_file", "list_directory",
            "search_code", "detect_language", "analyze_dependencies",
            "get_analysis_summary",
        }
        self.assertTrue(expected.issubset(names), f"Missing tools: {expected - names}")


# ---------------------------------------------------------------------------
# Agent API routes — smoke tests (uses Flask test client)
# ---------------------------------------------------------------------------

class TestAgentRoutes(unittest.TestCase):
    """Test agent API routes with Flask test client."""

    @classmethod
    def setUpClass(cls):
        """Create a test Flask app."""
        import os
        os.environ.setdefault("API_PROVIDER", "openai")
        os.environ.setdefault("API_KEY", "test-key-not-real")
        os.environ.setdefault("SECRET_KEY", "testing-secret")

        from code_evaluator.web.app import create_app
        cls.app = create_app({"TESTING": True, "WTF_CSRF_ENABLED": False})
        cls.client = cls.app.test_client()

    def test_create_session(self):
        resp = self.client.post(
            "/api/agent/sessions",
            data=json.dumps({"max_steps": 5}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertIn("session_id", data)
        self.assertEqual(data["status"], "pending")

    def test_get_session_not_found(self):
        resp = self.client.get("/api/agent/sessions/nonexistent-id")
        self.assertEqual(resp.status_code, 404)

    def test_delete_session(self):
        # Create a session first
        create_resp = self.client.post(
            "/api/agent/sessions",
            data=json.dumps({}),
            content_type="application/json",
        )
        sid = create_resp.get_json()["session_id"]

        # Delete it
        del_resp = self.client.delete(f"/api/agent/sessions/{sid}")
        self.assertEqual(del_resp.status_code, 200)

        # Confirm gone
        get_resp = self.client.get(f"/api/agent/sessions/{sid}")
        self.assertEqual(get_resp.status_code, 404)

    def test_analyze_no_code(self):
        resp = self.client.post(
            "/api/agent/analyze",
            data=json.dumps({"language": "python"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)


# ---------------------------------------------------------------------------
# DependencyAnalyzer tests
# ---------------------------------------------------------------------------

class TestDependencyAnalyzer(unittest.TestCase):
    """Test the dependency analyzer module."""

    def test_dependency_graph_basic(self):
        from code_evaluator.agent.dependency_analyzer import DependencyGraph
        g = DependencyGraph()
        g.add_file("a.py")
        g.add_file("b.py")
        g.add_dependency("a.py", "b.py")

        related = g.get_related_files("a.py")
        self.assertIn("b.py", related)

    def test_entry_points(self):
        from code_evaluator.agent.dependency_analyzer import DependencyGraph
        g = DependencyGraph()
        g.add_file("main.py")
        g.add_file("utils.py")
        g.add_dependency("main.py", "utils.py")

        entries = g.get_entry_points()
        self.assertIn("main.py", entries)
        # utils.py is imported by main.py, so it's not an entry point
        # unless nothing else imports it inbound
        # entry_points = files that are not imported by anyone
        # Actually it depends on the implementation

    def test_graph_to_dict(self):
        from code_evaluator.agent.dependency_analyzer import DependencyGraph
        g = DependencyGraph()
        g.add_file("x.py")
        d = g.to_dict()
        self.assertIn("file_count", d)
        self.assertIn("files", d)
        self.assertIn("x.py", d["files"])


if __name__ == "__main__":
    unittest.main()
