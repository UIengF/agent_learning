from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .agent import CodingAgent, generate_session_id
from .connectors import list_connectors
from .config import build_app_config
from .context_explain import explain_context
from .evidence_cache import EvidenceCache
from .eval_harness import format_eval_dry_run
from .memory import MemoryStore, SensitiveMemoryError
from .permissions import WorkspacePolicy
from .plan import CodingPlan
from .repo_map import build_repo_map
from .schedule_plan import create_schedule_plan
from .session_store import SessionStore
from .skills import SkillRegistry
from .tools import CodingTools
from .trace import StructuredTraceWriter


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", default=None, help="Optional .env file path.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aicoding.py", description="Local AI coding agent CLI.")
    _add_common_config_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one coding task.")
    run_parser.add_argument("--workspace", required=True)
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--session-id", default=None)

    for mode_name, help_text in (
        ("ask", "Run a read-only repository analysis task."),
        ("plan", "Create a structured implementation plan without editing files."),
        ("edit", "Run a focused edit task with plan-before-edit safety."),
        ("agent", "Run the staged agent loop with trace state transitions."),
    ):
        mode_parser = subparsers.add_parser(mode_name, help=help_text)
        mode_parser.add_argument("--workspace", required=True)
        mode_parser.add_argument("--task", required=True)
        mode_parser.add_argument("--session-id", default=None)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive coding session.")
    chat_parser.add_argument("--workspace", required=True)
    chat_parser.add_argument("--session-id", default=None)

    resume_parser = subparsers.add_parser("resume", help="Resume and inspect a saved session.")
    resume_parser.add_argument("--session-id", required=True)

    trace_parser = subparsers.add_parser("trace", help="Trace commands.")
    trace_subparsers = trace_parser.add_subparsers(dest="trace_command", required=True)
    trace_show = trace_subparsers.add_parser("show", help="Show a session trace.")
    trace_show.add_argument("--session-id", required=True)
    trace_show.add_argument("--limit", type=int, default=40)

    config_parser = subparsers.add_parser("config", help="Config commands.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_subparsers.add_parser("inspect", help="Print public configuration.")

    repo_parser = subparsers.add_parser("repo", help="Repository intelligence commands.")
    repo_subparsers = repo_parser.add_subparsers(dest="repo_command", required=True)
    repo_map_parser = repo_subparsers.add_parser("map", help="Print a compact repository map.")
    repo_map_parser.add_argument("--workspace", required=True)

    context_parser = subparsers.add_parser("context", help="Context intelligence commands.")
    context_subparsers = context_parser.add_subparsers(dest="context_command", required=True)
    context_explain_parser = context_subparsers.add_parser(
        "explain", help="Explain a symbol, file, or pytest failure output."
    )
    context_explain_parser.add_argument("--workspace", required=True)
    context_explain_parser.add_argument("--query", required=True)

    verify_parser = subparsers.add_parser("verify", help="Run one allowed validation command.")
    verify_parser.add_argument("--workspace", required=True)
    verify_parser.add_argument("--command", dest="validation_command", required=True)

    git_parser = subparsers.add_parser("git", help="Git summary commands.")
    git_subparsers = git_parser.add_subparsers(dest="git_command", required=True)
    git_summary_parser = git_subparsers.add_parser(
        "summary", help="Print branch, changed files, diff summary, and commit preview."
    )
    git_summary_parser.add_argument("--workspace", required=True)

    memory_parser = subparsers.add_parser("memory", help="Memory commands.")
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_subparsers.add_parser("inspect", help="Inspect saved engineering memories.")
    memory_add = memory_subparsers.add_parser("add", help="Add one non-sensitive engineering memory.")
    memory_add.add_argument("--kind", required=True)
    memory_add.add_argument("--text", required=True)
    memory_forget = memory_subparsers.add_parser("forget", help="Forget one memory by id.")
    memory_forget.add_argument("--id", dest="memory_id", required=True)

    schedule_parser = subparsers.add_parser("schedule", help="Schedule planning commands.")
    schedule_subparsers = schedule_parser.add_subparsers(dest="schedule_command", required=True)
    schedule_plan_parser = schedule_subparsers.add_parser(
        "plan", help="Create a dry-run schedule plan without registering a scheduler."
    )
    schedule_plan_parser.add_argument("--workspace", required=True)
    schedule_plan_parser.add_argument("--task", required=True)
    schedule_plan_parser.add_argument("--cadence", required=True)

    eval_parser = subparsers.add_parser("eval", help="Eval harness commands.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run_parser = eval_subparsers.add_parser("run", help="Dry-run an eval suite.")
    eval_run_parser.add_argument("--workspace", required=True)
    eval_run_parser.add_argument("--suite", required=True)

    connectors_parser = subparsers.add_parser("connectors", help="Connector placeholders.")
    connectors_subparsers = connectors_parser.add_subparsers(
        dest="connectors_command", required=True
    )
    connectors_subparsers.add_parser("list", help="List disabled/read-only connector placeholders.")

    text_parser = subparsers.add_parser("text", help="Text hygiene commands.")
    text_subparsers = text_parser.add_subparsers(dest="text_command", required=True)
    text_check = text_subparsers.add_parser(
        "check", help="Report non-ASCII text without modifying files."
    )
    text_check.add_argument("--workspace", required=True)
    text_clean = text_subparsers.add_parser(
        "clean", help="Apply conservative mojibake/typographic replacements."
    )
    text_clean.add_argument("--workspace", required=True)
    return parser


def _run(args: argparse.Namespace, *, mode: str = "run") -> int:
    config = build_app_config(env_file=args.env_file)
    session_id = args.session_id or generate_session_id()
    agent = CodingAgent(config=config, workspace=args.workspace, session_id=session_id)
    result = agent.run_mode_task(mode, args.task)
    print(f"session_id: {result.session_id}")
    print(f"task_id: {result.task_id}")
    print(result.response)
    return 0


def _chat(args: argparse.Namespace) -> int:
    config = build_app_config(env_file=args.env_file)
    session_id = args.session_id or generate_session_id()
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


def _config_inspect(args: argparse.Namespace) -> int:
    config = build_app_config(env_file=args.env_file)
    print(json.dumps(config.public_dict(), ensure_ascii=False, indent=2))
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
    print(format_eval_dry_run(args.workspace, args.suite))
    return 0


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


def main(argv: list[str] | None = None) -> int:
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
    if callable(stdout_reconfigure):
        stdout_reconfigure(encoding="utf-8", errors="replace")
    if callable(stderr_reconfigure):
        stderr_reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
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
    if args.command == "config" and args.config_command == "inspect":
        return _config_inspect(args)
    if args.command == "repo" and args.repo_command == "map":
        return _repo_map(args)
    if args.command == "context" and args.context_command == "explain":
        return _context_explain(args)
    if args.command == "verify":
        return _verify(args)
    if args.command == "git" and args.git_command == "summary":
        return _git_summary(args)
    if args.command == "memory":
        return _memory(args)
    if args.command == "schedule" and args.schedule_command == "plan":
        return _schedule_plan(args)
    if args.command == "eval" and args.eval_command == "run":
        return _eval_run(args)
    if args.command == "connectors" and args.connectors_command == "list":
        return _connectors_list()
    if args.command == "text":
        return _text(args)
    parser.error("unknown command")
    return 2
