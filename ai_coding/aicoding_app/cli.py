from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .agent import CodingAgent, generate_session_id
from .connectors import list_connectors
from .config import PROJECT_ROOT, build_app_config
from .context_explain import explain_context
from .evidence_cache import EvidenceCache
from .eval_harness import format_eval_result, run_eval_suite
from .memory import MemoryStore, SensitiveMemoryError
from .permissions import WorkspacePolicy
from .plan import CodingPlan
from .repo_map import build_repo_map
from .schedule_plan import create_schedule_plan
from .session_store import SessionStore
from .skills import SkillRegistry
from .tools import CodingTools
from .trace import StructuredTraceWriter
from .trace_viewer import serve_trace_viewer


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", default=None, help="Optional .env file path.")


def _progress(message: str) -> None:
    print(f"[aicoding] {message}", file=sys.stderr, flush=True)


def _add_workspace_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        required=True,
        help="Path to the repository or project workspace to operate on.",
    )


def _add_task_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, help="Natural-language task to run.")


def _add_session_id_arg(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument(
        "--session-id",
        required=required,
        default=None,
        help="Session id to create, reuse, inspect, or resume.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aicoding", description="Local AI coding agent CLI.")
    _add_common_config_args(parser)
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the installed aicoding version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one coding task.")
    _add_workspace_arg(run_parser)
    _add_task_arg(run_parser)
    _add_session_id_arg(run_parser)

    for mode_name, help_text in (
        ("ask", "Run a read-only repository analysis task."),
        ("plan", "Create a structured implementation plan without editing files."),
        ("edit", "Run a focused edit task with plan-before-edit safety."),
        ("agent", "Run the staged agent loop with trace state transitions."),
    ):
        mode_parser = subparsers.add_parser(mode_name, help=help_text)
        _add_workspace_arg(mode_parser)
        _add_task_arg(mode_parser)
        _add_session_id_arg(mode_parser)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive coding session.")
    _add_workspace_arg(chat_parser)
    _add_session_id_arg(chat_parser)

    resume_parser = subparsers.add_parser("resume", help="Resume and inspect a saved session.")
    _add_session_id_arg(resume_parser, required=True)

    trace_parser = subparsers.add_parser("trace", help="Trace commands.")
    trace_subparsers = trace_parser.add_subparsers(dest="trace_command", required=True)
    trace_show = trace_subparsers.add_parser("show", help="Show a session trace.")
    _add_session_id_arg(trace_show, required=True)
    trace_show.add_argument("--limit", type=int, default=40, help="Maximum trace lines to print.")
    trace_serve = trace_subparsers.add_parser("serve", help="Serve the local trace Web UI.")
    trace_serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    trace_serve.add_argument("--port", type=int, default=8765, help="Port to listen on.")
    trace_serve.add_argument(
        "--open", action="store_true", dest="open_browser", help="Open the viewer in a browser."
    )
    _add_session_id_arg(trace_serve)

    config_parser = subparsers.add_parser("config", help="Config commands.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_subparsers.add_parser("inspect", help="Print public configuration.")

    subparsers.add_parser("init", help="Create a local .env file from .env.example.")
    subparsers.add_parser("doctor", help="Print an environment and configuration health report.")

    dev_parser = subparsers.add_parser("dev", help="Developer diagnostics and maintenance commands.")
    dev_subparsers = dev_parser.add_subparsers(dest="dev_command", required=True)

    repo_parser = dev_subparsers.add_parser("repo", help="Repository intelligence commands.")
    repo_subparsers = repo_parser.add_subparsers(dest="repo_command", required=True)
    repo_map_parser = repo_subparsers.add_parser("map", help="Print a compact repository map.")
    _add_workspace_arg(repo_map_parser)

    context_parser = dev_subparsers.add_parser("context", help="Context intelligence commands.")
    context_subparsers = context_parser.add_subparsers(dest="context_command", required=True)
    context_explain_parser = context_subparsers.add_parser(
        "explain", help="Explain a symbol, file, or pytest failure output."
    )
    _add_workspace_arg(context_explain_parser)
    context_explain_parser.add_argument(
        "--query", required=True, help="Symbol, path, or failure output to explain."
    )

    verify_parser = dev_subparsers.add_parser("verify", help="Run one allowed validation command.")
    _add_workspace_arg(verify_parser)
    verify_parser.add_argument(
        "--validation-command",
        required=True,
        help="Allowed command to run in the workspace, such as 'python -m pytest tests'.",
    )

    git_parser = dev_subparsers.add_parser("git", help="Git summary commands.")
    git_subparsers = git_parser.add_subparsers(dest="git_command", required=True)
    git_summary_parser = git_subparsers.add_parser(
        "summary", help="Print branch, changed files, diff summary, and commit preview."
    )
    _add_workspace_arg(git_summary_parser)

    memory_parser = dev_subparsers.add_parser("memory", help="Memory commands.")
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_subparsers.add_parser("inspect", help="Inspect saved engineering memories.")
    memory_add = memory_subparsers.add_parser("add", help="Add one non-sensitive engineering memory.")
    memory_add.add_argument("--kind", required=True, help="Memory category, for example command.")
    memory_add.add_argument("--text", required=True, help="Non-sensitive memory text to store.")
    memory_forget = memory_subparsers.add_parser("forget", help="Forget one memory by id.")
    memory_forget.add_argument("--id", dest="memory_id", required=True, help="Memory id to delete.")

    schedule_parser = dev_subparsers.add_parser("schedule", help="Schedule planning commands.")
    schedule_subparsers = schedule_parser.add_subparsers(dest="schedule_command", required=True)
    schedule_plan_parser = schedule_subparsers.add_parser(
        "plan", help="Create a dry-run schedule plan without registering a scheduler."
    )
    _add_workspace_arg(schedule_plan_parser)
    _add_task_arg(schedule_plan_parser)
    schedule_plan_parser.add_argument(
        "--cadence", required=True, help="Schedule cadence, such as daily or weekly."
    )

    eval_parser = dev_subparsers.add_parser("eval", help="Eval harness commands.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run_parser = eval_subparsers.add_parser("run", help="Run an eval suite.")
    _add_workspace_arg(eval_run_parser)
    eval_run_parser.add_argument("--suite", required=True, help="Path to an eval suite JSON file.")

    connectors_parser = dev_subparsers.add_parser("connectors", help="Connector placeholders.")
    connectors_subparsers = connectors_parser.add_subparsers(
        dest="connectors_command", required=True
    )
    connectors_subparsers.add_parser("list", help="List disabled/read-only connector placeholders.")

    text_parser = dev_subparsers.add_parser("text", help="Text hygiene commands.")
    text_subparsers = text_parser.add_subparsers(dest="text_command", required=True)
    text_check = text_subparsers.add_parser(
        "check", help="Report non-ASCII text without modifying files."
    )
    _add_workspace_arg(text_check)
    text_clean = text_subparsers.add_parser(
        "clean", help="Apply conservative mojibake/typographic replacements."
    )
    _add_workspace_arg(text_clean)
    return parser


def _run(args: argparse.Namespace, *, mode: str = "run") -> int:
    config = build_app_config(env_file=args.env_file)
    session_id = args.session_id or generate_session_id()
    _progress(f"starting {mode} task in {args.workspace}")
    agent = CodingAgent(config=config, workspace=args.workspace, session_id=session_id)
    result = agent.run_mode_task(mode, args.task)
    print(f"session_id: {result.session_id}")
    print(f"task_id: {result.task_id}")
    print(result.response)
    return 0


def _chat(args: argparse.Namespace) -> int:
    config = build_app_config(env_file=args.env_file)
    session_id = args.session_id or generate_session_id()
    _progress(f"starting chat session in {args.workspace}")
    agent = CodingAgent(config=config, workspace=args.workspace, session_id=session_id)
    print(f"session_id: {session_id}")
    print("Type /exit to quit.")
    while True:
        try:
            task = input("> ").strip()
        except EOFError:
            break
        if not task:
            continue
        if task in {"/exit", "/quit"}:
            break
        result = agent.run_task(task)
        print(result.response)
    return 0


def _resume(args: argparse.Namespace) -> int:
    config = build_app_config(env_file=args.env_file)
    store = SessionStore(config.harness.runtime_dir)
    session = store.load(args.session_id)
    if session is None:
        print(f"session not found: {args.session_id}", file=sys.stderr)
        return 1
    print(json.dumps(session.to_jsonable(), ensure_ascii=False, indent=2))
    return 0


def _trace_show(args: argparse.Namespace) -> int:
    config = build_app_config(env_file=args.env_file)
    trace_path = config.harness.runtime_dir / "traces" / f"{args.session_id}.jsonl"
    if not trace_path.exists():
        print(f"trace not found: {args.session_id}", file=sys.stderr)
        return 1
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    for line in lines[-max(1, args.limit) :]:
        print(line)
    return 0


def _trace_serve(args: argparse.Namespace) -> int:
    config = build_app_config(env_file=args.env_file)
    serve_trace_viewer(
        config.harness.runtime_dir,
        app_config=config,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
        session_id=args.session_id,
    )
    return 0


def _config_inspect(args: argparse.Namespace) -> int:
    config = build_app_config(env_file=args.env_file)
    print(json.dumps(config.public_dict(), ensure_ascii=False, indent=2))
    return 0


def _init_env(_: argparse.Namespace) -> int:
    target = Path.cwd() / ".env"
    template = Path.cwd() / ".env.example"
    if not template.exists():
        template = PROJECT_ROOT / ".env.example"
    if target.exists():
        print(f".env already exists: {target}")
        print("Review AICODING_MODEL_API_KEY, AICODING_MODEL_NAME, and AICODING_MODEL_API_BASE.")
        return 0
    if not template.exists():
        print("missing template: .env.example", file=sys.stderr)
        return 1
    target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"created .env from {template}")
    print("Set required env vars before running agent tasks: AICODING_MODEL_API_KEY.")
    print("Optional model settings: AICODING_MODEL_NAME, AICODING_MODEL_API_BASE.")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    config = build_app_config(env_file=args.env_file)
    runtime_dir = config.harness.runtime_dir
    runtime_parent = runtime_dir if runtime_dir.exists() else runtime_dir.parent
    checks = [
        ("api_key", "ok" if config.model.api_key else "missing"),
        ("model_name", config.model.model_name or "missing"),
        ("api_base", config.model.api_base),
        ("runtime_dir", f"{runtime_dir} ({'exists' if runtime_dir.exists() else 'will be created'})"),
        (
            "runtime_parent",
            f"{runtime_parent} ({'ok' if runtime_parent.exists() else 'missing'})",
        ),
        (
            "python",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
            f"({'ok' if sys.version_info >= (3, 11) else 'requires >= 3.11'})",
        ),
    ]
    print("aicoding doctor")
    for name, value in checks:
        print(f"- {name}: {value}")
    return 0


def _repo_map(args: argparse.Namespace) -> int:
    print(build_repo_map(args.workspace).format())
    return 0


def _context_explain(args: argparse.Namespace) -> int:
    print(explain_context(args.workspace, args.query))
    return 0


def _tools_for_cli(args: argparse.Namespace, *, session_id: str) -> CodingTools:
    config = build_app_config(env_file=args.env_file)
    policy = WorkspacePolicy(
        workspace=Path(args.workspace),
        allowed_commands=config.harness.allowed_commands,
    )
    return CodingTools(
        policy=policy,
        plan=CodingPlan(),
        evidence_cache=EvidenceCache(),
        skill_registry=SkillRegistry(config.project_root / "skills"),
        trace_writer=StructuredTraceWriter(
            config.harness.runtime_dir / "traces",
            session_id,
            enabled=config.harness.trace_enabled,
        ),
        command_timeout_seconds=config.harness.command_timeout_seconds,
        task_id=session_id,
    )


def _verify(args: argparse.Namespace) -> int:
    tools = _tools_for_cli(args, session_id="verify")
    print(tools.run_validation(args.validation_command))
    return 0


def _git_summary(args: argparse.Namespace) -> int:
    tools = _tools_for_cli(args, session_id="git-summary")
    branch = tools.git_branch()
    changed = tools.git_diff_files()
    diff = tools.git_diff()
    preview = tools.git_commit_preview()
    print(
        "\n\n".join(
            [
                "Git summary:",
                "Branch:",
                branch,
                "Changed files:",
                changed,
                "Diff summary:",
                diff[:2000] or "No diff.",
                preview,
            ]
        )
    )
    return 0


def _memory_store(args: argparse.Namespace) -> MemoryStore:
    config = build_app_config(env_file=args.env_file)
    return MemoryStore(config.harness.runtime_dir)


def _memory(args: argparse.Namespace) -> int:
    store = _memory_store(args)
    if args.memory_command == "inspect":
        print(store.format())
        return 0
    if args.memory_command == "add":
        try:
            item = store.add(args.kind, args.text)
        except SensitiveMemoryError as exc:
            print(f"memory_rejected: {exc}", file=sys.stderr)
            return 1
        print(f"memory_added: {item.id}")
        return 0
    if args.memory_command == "forget":
        removed = store.forget(args.memory_id)
        print(f"memory_forgotten: {args.memory_id}" if removed else f"memory_not_found: {args.memory_id}")
        return 0 if removed else 1
    return 2


def _schedule_plan(args: argparse.Namespace) -> int:
    print(create_schedule_plan(args.workspace, args.task, args.cadence).format())
    return 0


def _eval_run(args: argparse.Namespace) -> int:
    config = build_app_config(env_file=args.env_file)
    result = run_eval_suite(config, args.workspace, args.suite)
    print(format_eval_result(result))
    return 1 if result.has_failures else 0


def _connectors_list() -> int:
    print(list_connectors())
    return 0


def _text(args: argparse.Namespace) -> int:
    tools = _tools_for_cli(args, session_id=f"text-{args.text_command}")
    if args.text_command == "check":
        print(tools.text_hygiene_check())
        return 0
    if args.text_command == "clean":
        print(tools.text_hygiene_clean())
        return 0
    return 2


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if hasattr(args, "workspace"):
        workspace = Path(args.workspace).expanduser()
        if not workspace.is_dir():
            parser.error(f"--workspace must be an existing directory: {args.workspace}")
        args.workspace = str(workspace.resolve())
    if hasattr(args, "task") and not args.task.strip():
        parser.error("--task must be non-empty")


def main(argv: list[str] | None = None) -> int:
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
    if callable(stdout_reconfigure):
        stdout_reconfigure(encoding="utf-8", errors="replace")
    if callable(stderr_reconfigure):
        stderr_reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.command == "run":
        return _run(args)
    if args.command in {"ask", "plan", "edit", "agent"}:
        return _run(args, mode=args.command)
    if args.command == "chat":
        return _chat(args)
    if args.command == "resume":
        return _resume(args)
    if args.command == "trace" and args.trace_command == "show":
        return _trace_show(args)
    if args.command == "trace" and args.trace_command == "serve":
        return _trace_serve(args)
    if args.command == "config" and args.config_command == "inspect":
        return _config_inspect(args)
    if args.command == "init":
        return _init_env(args)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "dev" and args.dev_command == "repo" and args.repo_command == "map":
        return _repo_map(args)
    if args.command == "dev" and args.dev_command == "context" and args.context_command == "explain":
        return _context_explain(args)
    if args.command == "dev" and args.dev_command == "verify":
        return _verify(args)
    if args.command == "dev" and args.dev_command == "git" and args.git_command == "summary":
        return _git_summary(args)
    if args.command == "dev" and args.dev_command == "memory":
        return _memory(args)
    if args.command == "dev" and args.dev_command == "schedule" and args.schedule_command == "plan":
        return _schedule_plan(args)
    if args.command == "dev" and args.dev_command == "eval" and args.eval_command == "run":
        return _eval_run(args)
    if args.command == "dev" and args.dev_command == "connectors" and args.connectors_command == "list":
        return _connectors_list()
    if args.command == "dev" and args.dev_command == "text":
        return _text(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
