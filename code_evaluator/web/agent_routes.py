"""
Agent API routes for the web application.
Provides REST endpoints and SSE streaming for the AI agent.
"""

import json
import time
import queue
import logging
import threading
from typing import Optional

from flask import (
    Blueprint,
    request,
    jsonify,
    Response,
    stream_with_context,
)

logger = logging.getLogger(__name__)

agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")

# ── Global singletons (initialized lazily) ──────────────────────────────
_executor = None
_session_manager = None
_lock = threading.Lock()


def _get_session_manager():
    """Get or create the global SessionManager."""
    global _session_manager
    if _session_manager is None:
        from code_evaluator.agent.session import SessionManager
        from code_evaluator.model.config import APIConfig

        config = APIConfig.from_env()
        _session_manager = SessionManager(ttl=config.agent_session_ttl)
    return _session_manager


def _get_executor():
    """Get or create the global AgentExecutor."""
    global _executor
    if _executor is None:
        with _lock:
            if _executor is None:
                from code_evaluator.agent.executor import AgentExecutor
                from code_evaluator.agent.tools import create_default_registry
                from code_evaluator.model.loader import ModelLoader
                from code_evaluator.model.config import APIConfig

                config = APIConfig.from_env()
                model_loader = ModelLoader(config=config)
                tool_registry = create_default_registry()
                session_manager = _get_session_manager()

                _executor = AgentExecutor(
                    model_loader=model_loader,
                    tool_registry=tool_registry,
                    session_manager=session_manager,
                )
    return _executor


# ── POST /api/agent/sessions — Create a new session ────────────────────
@agent_bp.route("/sessions", methods=["POST"])
def create_session():
    """Create a new agent session."""
    data = request.get_json(silent=True) or {}
    max_steps = min(data.get("max_steps", 15), 30)  # Cap at 30

    sm = _get_session_manager()
    session = sm.create(max_steps=max_steps)

    return jsonify({
        "session_id": session.session_id,
        "status": session.status.value,
        "max_steps": session.max_steps,
    }), 201


# ── POST /api/agent/sessions/<id>/messages — Send message ──────────────
@agent_bp.route("/sessions/<session_id>/messages", methods=["POST"])
def send_message(session_id: str):
    """Send a message to an agent session and run the agent."""
    sm = _get_session_manager()
    session = sm.get(session_id)
    if session is None:
        return jsonify({"error": "Session not found."}), 404

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    message = data.get("message", "").strip()
    code = data.get("code", "").strip()
    language = data.get("language", "auto").strip()

    if not message and not code:
        return jsonify({"error": "Provide 'message' or 'code'."}), 400

    if not message:
        message = f"Please analyze this {language} code."

    executor = _get_executor()

    # Run agent in background thread
    def _run():
        try:
            if session.current_step == 0:
                # Fresh session
                for _ in executor.run(session, message, code=code or None, language=language):
                    pass
            else:
                # Continuation
                for _ in executor.continue_conversation(session, message):
                    pass
        except Exception as e:
            logger.error(f"Agent error in session {session_id}: {e}", exc_info=True)
            session.set_error(str(e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({
        "session_id": session.session_id,
        "status": "running",
        "message": "Agent is processing your request.",
    }), 202


# ── GET /api/agent/sessions/<id>/stream — SSE stream ───────────────────
@agent_bp.route("/sessions/<session_id>/stream", methods=["GET"])
def stream_session(session_id: str):
    """Stream agent steps via Server-Sent Events (SSE)."""
    sm = _get_session_manager()
    session = sm.get(session_id)
    if session is None:
        return jsonify({"error": "Session not found."}), 404

    def event_stream():
        step_queue = queue.Queue()
        last_step_seen = 0

        # Send already-completed steps first
        for step in session.steps:
            if step.step_number > last_step_seen:
                last_step_seen = step.step_number
                yield f"data: {json.dumps(step.to_dict())}\n\n"

        # If session already done, send completion and stop
        if session.status.value in ("completed", "error"):
            yield f"data: {json.dumps({'type': 'session_end', 'status': session.status.value, 'result': session.result})}\n\n"
            return

        # Subscribe to new steps
        def on_step(step):
            step_queue.put(step)

        session.subscribe(on_step)

        try:
            timeout_at = time.time() + 300  # 5 minute timeout
            while time.time() < timeout_at:
                try:
                    step = step_queue.get(timeout=1.0)
                    if step.step_number > last_step_seen:
                        last_step_seen = step.step_number
                        yield f"data: {json.dumps(step.to_dict())}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield f": keepalive\n\n"

                # Check if session is done
                if session.status.value in ("completed", "error"):
                    yield f"data: {json.dumps({'type': 'session_end', 'status': session.status.value, 'result': session.result})}\n\n"
                    return
        finally:
            session.unsubscribe(on_step)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── GET /api/agent/sessions/<id> — Get session status ──────────────────
@agent_bp.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """Get the current state of an agent session."""
    sm = _get_session_manager()
    session = sm.get(session_id)
    if session is None:
        return jsonify({"error": "Session not found."}), 404

    return jsonify(session.to_dict())


# ── GET /api/agent/sessions/<id>/steps — Get all steps ─────────────────
@agent_bp.route("/sessions/<session_id>/steps", methods=["GET"])
def get_session_steps(session_id: str):
    """Get all steps of an agent session."""
    sm = _get_session_manager()
    session = sm.get(session_id)
    if session is None:
        return jsonify({"error": "Session not found."}), 404

    return jsonify({
        "session_id": session_id,
        "step_count": len(session.steps),
        "steps": [s.to_dict() for s in session.steps],
    })


# ── DELETE /api/agent/sessions/<id> — Delete session ───────────────────
@agent_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Delete an agent session."""
    sm = _get_session_manager()
    if sm.delete(session_id):
        return jsonify({"message": "Session deleted."}), 200
    return jsonify({"error": "Session not found."}), 404


# ── POST /api/agent/analyze — One-shot agent analysis ──────────────────
@agent_bp.route("/analyze", methods=["POST"])
def agent_analyze():
    """
    Synchronous one-shot agent analysis.
    Creates a session, runs full analysis, returns results.
    Expects: { "code": "...", "language": "auto|cpp|python|..." }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    code = data.get("code", "").strip()
    language = data.get("language", "auto").strip()
    max_steps = min(data.get("max_steps", 15), 30)

    if not code:
        return jsonify({"error": "No code provided."}), 400

    # Validate code content
    from code_evaluator.web.validators import validate_code_content
    is_valid, error_msg = validate_code_content(code)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    executor = _get_executor()

    try:
        session = executor.run_analysis(
            code=code,
            language=language,
            max_steps=max_steps,
        )

        # Build response in the same format as /api/analyze for compatibility
        result = session.result or {}

        # If the result has the expected structure, return it directly
        issues = result.get("issues", [])
        suggested_fixes = result.get("suggested_fixes", [])

        return jsonify({
            "language": result.get("language", language),
            "summary": result.get("summary", ""),
            "overall_score": result.get("overall_score", 0),
            "issues": issues,
            "suggested_fixes": suggested_fixes,
            "agent_session": {
                "session_id": session.session_id,
                "steps": [s.to_dict() for s in session.steps],
                "context": session.context,
            },
        })
    except Exception as e:
        logger.error(f"Agent analyze error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── POST /api/agent/project — Project analysis ─────────────────────────
@agent_bp.route("/project", methods=["POST"])
def agent_project_analyze():
    """
    Analyze an entire project directory.
    Expects: { "directory": "/path/to/project" }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    directory = data.get("directory", "").strip()
    max_steps = min(data.get("max_steps", 25), 40)

    if not directory:
        return jsonify({"error": "No directory provided."}), 400

    from code_evaluator.utils.security import validate_path
    is_valid, error_msg = validate_path(directory)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    executor = _get_executor()

    try:
        session = executor.run_project_analysis(
            directory=directory,
            max_steps=max_steps,
        )
        return jsonify(session.to_dict())
    except Exception as e:
        logger.error(f"Agent project analyze error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
