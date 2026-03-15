#!/usr/bin/env python
"""
Code Evaluator - Main CLI Entry Point
A multi-language code analyzer using LLM API providers

Usage:
    python main.py analyze [files...]       - Analyze code files
    python main.py agent analyze [files...] - Agent-based analysis (multi-step)
    python main.py agent chat               - Interactive agent chat
    python main.py agent project <dir>      - Agent-based project analysis
    python main.py serve                    - Start web server
"""

import argparse
import os
import sys
import logging

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_analyze(args):
    """Handle analyze command"""
    from code_evaluator import CodeAnalyzer, generate_report, save_results
    from code_evaluator.model.config import APIConfig

    # Handle case when no files are provided
    if not args.files:
        sample_path = "examples/example.cpp"
        if os.path.exists(sample_path):
            args.files = [sample_path]
            print(f"[INFO] No files provided, using sample file: {sample_path}")
        else:
            print("[ERROR] No files provided and no sample file found.")
            sys.exit(1)

    # Build API config from CLI args or environment
    config = APIConfig.from_env()
    if args.provider:
        config.provider = args.provider
    if args.api_key:
        config.api_key = args.api_key
    if args.api_model:
        config.model = args.api_model

    # Initialize analyzer
    analyzer = CodeAnalyzer(config=config)
    analyzer.load_model()

    # Process each file
    all_results = []
    for file_path in args.files:
        if args.verbose:
            print(f"[INFO] Analyzing {file_path}...")

        # Analyze the file
        results = analyzer.analyze_file(file_path)

        all_results.append(results)

        # Save individual results if output directory is specified
        if args.output:
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_analysis.json")
            save_results(results, output_path)

        # Generate and save individual report if report directory is specified
        report = generate_report(results)
        if args.report:
            report_dir = args.report
            os.makedirs(report_dir, exist_ok=True)
            base_name = os.path.basename(file_path)
            report_path = os.path.join(report_dir, f"{os.path.splitext(base_name)[0]}_report.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            if args.verbose:
                print(f"[INFO] Report saved to {report_path}")
        else:
            print(report)
            print("\n" + "-" * 80 + "\n")

    # If multiple files were analyzed, save a summary
    if len(all_results) > 1 and args.output:
        from code_evaluator.report.exporter import save_summary
        summary_path = os.path.join(args.output, "analysis_summary.json")
        save_summary(all_results, summary_path)
        if args.verbose:
            print(f"[INFO] Summary saved to {summary_path}")


def cmd_serve(args):
    """Handle serve command"""
    from code_evaluator.web import create_app

    app = create_app()
    
    try:
        port = int(os.environ.get('PORT', args.port))
    except ValueError:
        logging.warning(f"Invalid PORT value. Using default port {args.port}.")
        port = args.port
    
    print(f"[INFO] Starting web server on http://{args.host}:{port}")
    app.run(debug=args.debug, host=args.host, port=port)


# ── Agent CLI commands ──────────────────────────────────────────────────

def _create_agent_executor(config=None):
    """Shared helper to build an AgentExecutor from config."""
    from code_evaluator.agent.executor import AgentExecutor
    from code_evaluator.agent.tools import create_default_registry
    from code_evaluator.agent.session import SessionManager
    from code_evaluator.model.loader import ModelLoader
    from code_evaluator.model.config import APIConfig

    if config is None:
        config = APIConfig.from_env()

    model_loader = ModelLoader(config=config)
    tool_registry = create_default_registry()
    session_manager = SessionManager(ttl=config.agent_session_ttl)

    return AgentExecutor(
        model_loader=model_loader,
        tool_registry=tool_registry,
        session_manager=session_manager,
    )


def _print_step(step, verbose=False):
    """Pretty-print an agent step to the console."""
    from code_evaluator.agent.session import StepType

    icons = {
        StepType.THINKING: "[THINK]",
        StepType.TOOL_CALL: "[TOOL ]",
        StepType.TOOL_RESULT: "[RSLT ]",
        StepType.RESPONSE: "[REPLY]",
        StepType.ERROR: "[ERROR]",
    }
    icon = icons.get(step.type, "[????]")

    if step.type == StepType.THINKING:
        print(f"  {icon} {step.content}")
    elif step.type == StepType.TOOL_CALL:
        args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in step.tool_args.items())
        print(f"  {icon} {step.tool_name}({args_str})")
    elif step.type == StepType.TOOL_RESULT:
        truncated = step.tool_result[:200] + "..." if len(step.tool_result) > 200 else step.tool_result
        if verbose:
            print(f"  {icon} {truncated}")
    elif step.type == StepType.RESPONSE:
        print(f"\n{icon} Agent Response:\n{'=' * 60}")
        print(step.content)
        print("=" * 60)
    elif step.type == StepType.ERROR:
        print(f"  {icon} {step.content}")


def cmd_agent_analyze(args):
    """Handle agent analyze command — multi-step agent analysis of files."""
    from code_evaluator.model.config import APIConfig

    if not args.files:
        sample_path = "examples/example.cpp"
        if os.path.exists(sample_path):
            args.files = [sample_path]
            print(f"[INFO] No files provided, using sample file: {sample_path}")
        else:
            print("[ERROR] No files provided and no sample file found.")
            sys.exit(1)

    config = APIConfig.from_env()
    if args.provider:
        config.provider = args.provider
    if args.api_key:
        config.api_key = args.api_key
    if args.api_model:
        config.model = args.api_model

    executor = _create_agent_executor(config)
    max_steps = args.max_steps or config.agent_max_steps

    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Agent Analysis: {file_path}")
        print(f"{'=' * 60}\n")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except (IOError, OSError) as e:
            print(f"[ERROR] Cannot read file: {e}")
            continue

        from code_evaluator.utils.file_utils import detect_language
        language = detect_language(file_path)

        session = executor.session_manager.create(max_steps=max_steps)
        user_msg = f"Please perform a comprehensive analysis of this {language} code from file: {file_path}"

        for step in executor.run(session, user_msg, code=code, language=language):
            _print_step(step, verbose=args.verbose)

        # Show summary
        if session.result:
            score = session.result.get("overall_score", "N/A")
            issues = session.context.get("issues_found", 0)
            fixes = session.context.get("fixes_applied", 0)
            print(f"\n[SUMMARY] Score: {score}/100 | Issues: {issues} | Fixes applied: {fixes}")

        # Save results
        if args.output:
            import json
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.basename(file_path)
            output_path = os.path.join(args.output, f"{os.path.splitext(base_name)[0]}_agent.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
            print(f"[INFO] Results saved to {output_path}")


def cmd_agent_project(args):
    """Handle agent project command — multi-step directory analysis."""
    from code_evaluator.model.config import APIConfig

    directory = args.directory
    if not os.path.isdir(directory):
        print(f"[ERROR] Not a directory: {directory}")
        sys.exit(1)

    config = APIConfig.from_env()
    if args.provider:
        config.provider = args.provider
    if args.api_key:
        config.api_key = args.api_key

    executor = _create_agent_executor(config)
    max_steps = args.max_steps or 25

    print(f"\n{'=' * 60}")
    print(f"  Agent Project Analysis: {directory}")
    print(f"{'=' * 60}\n")

    session = executor.session_manager.create(max_steps=max_steps)
    user_msg = (
        f"Please analyze the project in directory: {directory}\n"
        "Start by listing the directory structure, then read and analyze "
        "the most important source files. Look for cross-file issues, "
        "architectural problems, and provide a comprehensive report."
    )

    for step in executor.run(session, user_msg):
        _print_step(step, verbose=args.verbose)

    if args.output:
        import json
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "project_agent_analysis.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2)
        print(f"[INFO] Results saved to {output_path}")


def cmd_agent_chat(args):
    """Handle agent chat command — interactive agent conversation."""
    from code_evaluator.model.config import APIConfig

    config = APIConfig.from_env()
    if args.provider:
        config.provider = args.provider
    if args.api_key:
        config.api_key = args.api_key

    executor = _create_agent_executor(config)
    session = executor.session_manager.create(max_steps=args.max_steps or 50)

    print("=" * 60)
    print("  Code Evaluator — Interactive Agent Chat")
    print("  Type 'exit' or 'quit' to end. Type 'reset' for new session.")
    print("=" * 60)

    first_turn = True

    while True:
        try:
            user_input = input("\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("[INFO] Goodbye!")
            break
        if user_input.lower() == "reset":
            session = executor.session_manager.create(max_steps=args.max_steps or 50)
            first_turn = True
            print("[INFO] Session reset.")
            continue

        if first_turn:
            for step in executor.run(session, user_input):
                _print_step(step, verbose=args.verbose)
            first_turn = False
        else:
            for step in executor.continue_conversation(session, user_input):
                _print_step(step, verbose=args.verbose)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Code Evaluator - Multi-language code analyzer using LLM API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ============ Analyze command ============
    analyze_parser = subparsers.add_parser("analyze", help="Analyze code files")
    analyze_parser.add_argument("files", nargs="*", help="Path to code file(s) to analyze")
    analyze_parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"],
                                help="API provider (overrides API_PROVIDER env var)")
    analyze_parser.add_argument("--api-key", help="API key (overrides API_KEY env var)")
    analyze_parser.add_argument("--api-model", help="Model name (overrides API_MODEL env var)")
    analyze_parser.add_argument("--output", help="Directory to save analysis results (JSON)")
    analyze_parser.add_argument("--report", help="Directory to save human-readable reports (Markdown)")
    analyze_parser.add_argument("--no-cache", action="store_true", help="Disable caching of analysis results")
    analyze_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    analyze_parser.set_defaults(func=cmd_analyze)

    # ============ Serve command ============
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    serve_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    serve_parser.set_defaults(func=cmd_serve)

    # ============ Agent command ============
    agent_parser = subparsers.add_parser("agent", help="AI agent-based analysis (multi-step reasoning)")
    agent_sub = agent_parser.add_subparsers(dest="agent_command", help="Agent sub-commands")

    # -- agent analyze <files>
    aa_parser = agent_sub.add_parser("analyze", help="Agent-based file analysis")
    aa_parser.add_argument("files", nargs="*", help="Code file(s) to analyze")
    aa_parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"],
                           help="API provider")
    aa_parser.add_argument("--api-key", help="API key")
    aa_parser.add_argument("--api-model", help="Model name")
    aa_parser.add_argument("--max-steps", type=int, help="Max agent steps (default: 15)")
    aa_parser.add_argument("--output", help="Directory to save agent results (JSON)")
    aa_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    aa_parser.set_defaults(func=cmd_agent_analyze)

    # -- agent project <directory>
    ap_parser = agent_sub.add_parser("project", help="Agent-based project/directory analysis")
    ap_parser.add_argument("directory", help="Project directory to analyze")
    ap_parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"],
                           help="API provider")
    ap_parser.add_argument("--api-key", help="API key")
    ap_parser.add_argument("--max-steps", type=int, help="Max agent steps (default: 25)")
    ap_parser.add_argument("--output", help="Directory to save results (JSON)")
    ap_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    ap_parser.set_defaults(func=cmd_agent_project)

    # -- agent chat
    ac_parser = agent_sub.add_parser("chat", help="Interactive agent conversation")
    ac_parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"],
                           help="API provider")
    ac_parser.add_argument("--api-key", help="API key")
    ac_parser.add_argument("--max-steps", type=int, help="Max steps per turn (default: 50)")
    ac_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    ac_parser.set_defaults(func=cmd_agent_chat)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # For agent command without subcommand, print help
    if args.command == "agent" and not hasattr(args, "func"):
        agent_parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
