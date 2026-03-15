"""
Agent Executor — Core ReAct Loop
Implements the Reason → Act → Observe loop for the AI agent.
Orchestrates LLM calls, tool execution, and step tracking.
"""

import os
import json
import time
import logging
from typing import Any, Dict, Generator, List, Optional

from code_evaluator.model.loader import ModelLoader
from code_evaluator.model.config import APIConfig
from code_evaluator.model.base_client import ChatResponse
from code_evaluator.agent.tool_registry import ToolRegistry
from code_evaluator.agent.session import (
    AgentSession,
    AgentStep,
    SessionManager,
    SessionStatus,
    StepType,
)

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Executes the AI agent's ReAct loop:
    1. Send messages + tools to LLM
    2. If LLM returns tool_calls → execute tools → append results → go to 1
    3. If LLM returns text → finalize as the agent's response
    4. Repeat until done or max_steps reached
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        system_prompt: Optional[str] = None,
    ):
        self.model_loader = model_loader
        self.tool_registry = tool_registry
        self.session_manager = session_manager
        self._system_prompt = system_prompt or self._load_default_prompt()

    @staticmethod
    def _load_default_prompt() -> str:
        """Load the agent system prompt from file."""
        prompts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..",
            "prompts",
        )
        prompts_dir = os.path.normpath(prompts_dir)
        prompt_path = os.path.join(prompts_dir, "agent_system.txt")

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Agent system prompt not found at {prompt_path}, using fallback.")
            return _FALLBACK_SYSTEM_PROMPT

    def run(
        self,
        session: AgentSession,
        user_message: str,
        code: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Generator[AgentStep, None, None]:
        """
        Run the agent loop on a session.

        Args:
            session: The agent session to execute within.
            user_message: The user's message / request.
            code: Optional code to analyze.
            language: Optional language of the code.

        Yields:
            AgentStep for each action taken.
        """
        session.set_running()

        # Ensure model is loaded
        if not self.model_loader.is_loaded:
            if not self.model_loader.load():
                step = AgentStep(
                    step_number=1,
                    type=StepType.ERROR,
                    content="Failed to initialize LLM API client.",
                )
                session.add_step(step)
                session.set_error("API client initialization failed.")
                yield step
                return

        # Build initial messages
        system_msg = self._system_prompt
        if code:
            system_msg += f"\n\nThe user has provided the following code for analysis:\n```{language or ''}\n{code}\n```"

        session.add_message("system", system_msg)
        session.add_message("user", user_message)

        # Get tool definitions
        tool_defs = self.tool_registry.get_tool_definitions()

        step_num = 0

        # ReAct loop
        while session.can_continue:
            step_num += 1

            # ── Reason: call LLM with messages + tools ──
            thinking_step = AgentStep(
                step_number=step_num,
                type=StepType.THINKING,
                content="Analyzing and deciding next action...",
            )
            session.add_step(thinking_step)
            yield thinking_step

            try:
                response = self.model_loader.analyze(
                    messages=session.messages,
                    json_mode=False,
                    tools=tool_defs,
                    tool_choice="auto",
                )
            except Exception as e:
                error_step = AgentStep(
                    step_number=step_num,
                    type=StepType.ERROR,
                    content=f"LLM API error: {str(e)}",
                )
                session.add_step(error_step)
                session.set_error(str(e))
                yield error_step
                return

            # ── Handle response ──
            if isinstance(response, ChatResponse):
                if response.has_tool_calls:
                    # Store assistant message with tool calls in conversation
                    # For OpenAI: need to include the tool_calls in the assistant message
                    if response.raw:
                        session.add_message(**response.raw)
                    else:
                        session.add_message("assistant", response.content or "")

                    # ── Act: execute each tool call ──
                    for tc in response.tool_calls:
                        step_num += 1

                        # Record tool call step
                        tool_step = AgentStep(
                            step_number=step_num,
                            type=StepType.TOOL_CALL,
                            content=f"Calling tool: {tc.name}",
                            tool_name=tc.name,
                            tool_args=tc.arguments,
                        )
                        session.add_step(tool_step)
                        yield tool_step

                        # Execute tool
                        try:
                            result = self.tool_registry.execute(tc.name, tc.arguments)
                        except Exception as e:
                            result = json.dumps({"error": f"Tool execution failed: {str(e)}"})

                        step_num += 1

                        # ── Observe: record result ──
                        result_step = AgentStep(
                            step_number=step_num,
                            type=StepType.TOOL_RESULT,
                            content=f"Result from {tc.name}",
                            tool_name=tc.name,
                            tool_result=result,
                        )
                        session.add_step(result_step)
                        yield result_step

                        # Update session context
                        self._update_context(session, tc.name, result)

                        # Add tool result to conversation
                        session.add_message(
                            "tool",
                            result,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )

                    # Continue loop — LLM will see tool results
                    continue

                else:
                    # LLM returned text response (no more tool calls) → done
                    final_content = response.content
            elif isinstance(response, str):
                final_content = response
            else:
                final_content = str(response)

            # ── Final response step ──
            step_num += 1
            final_step = AgentStep(
                step_number=step_num,
                type=StepType.RESPONSE,
                content=final_content,
            )
            session.add_step(final_step)
            session.add_message("assistant", final_content)
            yield final_step

            # Parse and store final result
            result = self._parse_final_response(final_content, session)
            session.set_completed(result)
            return

        # Max steps reached
        max_step = AgentStep(
            step_number=step_num + 1,
            type=StepType.ERROR,
            content=f"Agent reached maximum step limit ({session.max_steps}). Returning partial results.",
        )
        session.add_step(max_step)
        yield max_step

        # Try to compile partial results
        result = self._compile_partial_results(session)
        session.set_completed(result)

    def run_analysis(
        self,
        code: str,
        language: str = "auto",
        file_path: Optional[str] = None,
        max_steps: int = 15,
    ) -> AgentSession:
        """
        Convenience method: create a session and run full code analysis.

        Args:
            code: Source code to analyze.
            language: Programming language (or 'auto').
            file_path: Optional file path for context.
            max_steps: Maximum agent steps.

        Returns:
            Completed AgentSession with results.
        """
        session = self.session_manager.create(max_steps=max_steps)

        user_msg = f"Please perform a comprehensive analysis of this {language} code."
        if file_path:
            user_msg += f" (from file: {file_path})"
            session.context["files_analyzed"].append(file_path)

        # Run all steps (consume generator)
        for step in self.run(session, user_msg, code=code, language=language):
            logger.debug(f"Step {step.step_number}: {step.type.value} - {step.content[:100]}")

        return session

    def run_project_analysis(
        self,
        directory: str,
        max_steps: int = 25,
    ) -> AgentSession:
        """
        Analyze an entire project directory.

        Args:
            directory: Path to the project directory.
            max_steps: Maximum agent steps.

        Returns:
            Completed AgentSession.
        """
        session = self.session_manager.create(max_steps=max_steps)

        user_msg = (
            f"Please analyze the project in directory: {directory}\n"
            "Start by listing the directory structure, then read and analyze "
            "the most important source files. Look for cross-file issues, "
            "architectural problems, and provide a comprehensive report."
        )

        for step in self.run(session, user_msg):
            logger.debug(f"Step {step.step_number}: {step.type.value}")

        return session

    def continue_conversation(
        self,
        session: AgentSession,
        user_message: str,
    ) -> Generator[AgentStep, None, None]:
        """
        Continue an existing conversation session.

        Args:
            session: Existing session to continue.
            user_message: Follow-up message from the user.

        Yields:
            AgentStep for each action taken.
        """
        # Reset session for continuation
        session.status = SessionStatus.RUNNING
        session.add_message("user", user_message)

        tool_defs = self.tool_registry.get_tool_definitions()
        step_num = session.current_step

        while session.can_continue:
            step_num += 1

            thinking_step = AgentStep(
                step_number=step_num,
                type=StepType.THINKING,
                content="Processing follow-up...",
            )
            session.add_step(thinking_step)
            yield thinking_step

            try:
                response = self.model_loader.analyze(
                    messages=session.messages,
                    json_mode=False,
                    tools=tool_defs,
                    tool_choice="auto",
                )
            except Exception as e:
                error_step = AgentStep(
                    step_number=step_num,
                    type=StepType.ERROR,
                    content=f"LLM API error: {str(e)}",
                )
                session.add_step(error_step)
                session.set_error(str(e))
                yield error_step
                return

            if isinstance(response, ChatResponse) and response.has_tool_calls:
                if response.raw:
                    session.add_message(**response.raw)
                else:
                    session.add_message("assistant", response.content or "")

                for tc in response.tool_calls:
                    step_num += 1
                    tool_step = AgentStep(
                        step_number=step_num,
                        type=StepType.TOOL_CALL,
                        content=f"Calling tool: {tc.name}",
                        tool_name=tc.name,
                        tool_args=tc.arguments,
                    )
                    session.add_step(tool_step)
                    yield tool_step

                    try:
                        result = self.tool_registry.execute(tc.name, tc.arguments)
                    except Exception as e:
                        result = json.dumps({"error": str(e)})

                    step_num += 1
                    result_step = AgentStep(
                        step_number=step_num,
                        type=StepType.TOOL_RESULT,
                        content=f"Result from {tc.name}",
                        tool_name=tc.name,
                        tool_result=result,
                    )
                    session.add_step(result_step)
                    yield result_step

                    self._update_context(session, tc.name, result)
                    session.add_message("tool", result, tool_call_id=tc.id, name=tc.name)

                continue
            else:
                content = response.content if isinstance(response, ChatResponse) else str(response)
                step_num += 1
                final_step = AgentStep(
                    step_number=step_num,
                    type=StepType.RESPONSE,
                    content=content,
                )
                session.add_step(final_step)
                session.add_message("assistant", content)
                yield final_step
                session.set_completed()
                return

        # Max steps
        session.set_completed()

    @staticmethod
    def _update_context(session: AgentSession, tool_name: str, result: str) -> None:
        """Update session context based on tool results."""
        try:
            data = json.loads(result) if isinstance(result, str) else result
        except (json.JSONDecodeError, TypeError):
            return

        if tool_name == "check_syntax" and isinstance(data, dict):
            errors = data.get("errors", [])
            session.context["issues_found"] += len(errors)

        elif tool_name == "analyze_patterns" and isinstance(data, dict):
            issues = data.get("issues", [])
            session.context["issues_found"] += len(issues)

        elif tool_name == "apply_fix" and isinstance(data, dict):
            if data.get("status") == "fix_applied":
                session.context["fixes_applied"] += 1

        elif tool_name == "verify_fix" and isinstance(data, dict):
            if data.get("status") == "verified":
                session.context["fixes_verified"] += 1

        elif tool_name == "read_file" and isinstance(data, dict):
            fp = data.get("file_path", "")
            if fp and fp not in session.context.get("files_analyzed", []):
                session.context.setdefault("files_analyzed", []).append(fp)

    @staticmethod
    def _parse_final_response(content: str, session: AgentSession) -> Dict:
        """Try to parse the agent's final response into structured results."""
        # Try JSON parsing
        try:
            # Try to extract JSON from the response
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n", 1)
                if len(lines) > 1:
                    text = lines[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = text[start:end + 1]
                parsed = json.loads(json_text)
                if isinstance(parsed, dict):
                    parsed["agent_context"] = session.context
                    return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Return as structured text result
        return {
            "summary": content[:500] if len(content) > 500 else content,
            "full_response": content,
            "agent_context": session.context,
        }

    @staticmethod
    def _compile_partial_results(session: AgentSession) -> Dict:
        """Compile partial results from session steps when max steps is reached."""
        tool_results = []
        last_response = ""

        for step in session.steps:
            if step.type == StepType.TOOL_RESULT and step.tool_result:
                try:
                    data = json.loads(step.tool_result)
                    tool_results.append({"tool": step.tool_name, "result": data})
                except (json.JSONDecodeError, TypeError):
                    tool_results.append({"tool": step.tool_name, "result": step.tool_result})

            if step.type == StepType.RESPONSE:
                last_response = step.content

        return {
            "summary": last_response or "Analysis reached step limit. Partial results below.",
            "partial": True,
            "tool_results": tool_results,
            "agent_context": session.context,
        }


_FALLBACK_SYSTEM_PROMPT = """You are an expert AI code analysis agent. You have access to tools that help you analyze code thoroughly.

Your approach should be:
1. First understand the code structure and language
2. Run syntax checking to find compilation/interpretation errors
3. Analyze patterns for common issues
4. Identify bugs, memory issues, security vulnerabilities, performance problems, and style issues
5. Suggest fixes for critical and high severity issues
6. Apply and verify fixes when possible
7. Provide a comprehensive final report

Use the available tools strategically. Don't try to do everything at once — work step by step.
When you have gathered enough information, provide your final analysis as a comprehensive JSON report with:
- language: detected programming language
- summary: brief overall summary
- overall_score: quality score from 0-100
- issues: array of {line, category, severity, description, recommendation}
- suggested_fixes: array of {line, original, fixed, explanation}

Always be thorough but efficient. Focus on the most impactful issues first."""
