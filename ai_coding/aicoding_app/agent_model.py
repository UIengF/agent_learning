from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent_progress import progress as _progress

try:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    CHAT_MODEL_AVAILABLE = True
    CHAT_MODEL_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover
    HumanMessage = None
    SystemMessage = None
    ToolMessage = None
    ChatOpenAI = None
    CHAT_MODEL_AVAILABLE = False
    CHAT_MODEL_IMPORT_ERROR = str(exc)

try:
    from langchain_core.tools import StructuredTool
    from langgraph.graph import END, MessagesState, StateGraph

    LANGGRAPH_AVAILABLE = CHAT_MODEL_AVAILABLE
    LANGGRAPH_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover
    StructuredTool = None
    MessagesState = None
    StateGraph = None
    END = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = str(exc)


PROMPT = """\
You are an AI coding agent operating inside one local workspace.

Use tools to inspect files before editing. Before any apply_patch call, call
plan_update with a concrete goal, edit steps, validation steps, and risks.
Before apply_patch, call preview_patch with the exact same patch and only apply
after the preview succeeds.
For long complete files such as README files or large test modules, prefer
write_text_file(path, content). It is a harness-controlled, traceable text file
write tool and must also be preceded by plan_update.
When adding documentation for a specific file, module, or feature, inspect the
target documentation first and verify it covers the same topic. If an existing
README clearly documents a different project, module, or domain, create a
dedicated nearby document instead of appending unrelated content.
If validation reports that Python, pytest, ruff, or pyright failed because the
validation environment itself is broken, treat it as an environment/tooling
blocker rather than an ordinary code failure. Do not retry the same failing validation command.
Call check_environment once to diagnose fixed Python validation tooling.
Do not automatically use install_python_package unless the task or configuration
explicitly asks you to repair the environment. Do not use
python -c, pip, or shell control operators to work around the command whitelist.
Do not call check_environment repeatedly. After one diagnosis, continue with
other allowed smoke tests, ruff, pyright, or report the
environment limitation clearly.
Never access paths outside the workspace. Only run commands through run_command.
Commands already run with the workspace as current directory. Do not prefix
commands with directory changes such as `cd`, `cd ..`, or `cd <path>`. Do not
combine commands with `&&`, `||`, `;`, or pipes. If a command is denied, simplify
to one bare allowed command such as `python script.py ...`, `python -m pytest ...`,
`python -m ruff check ...`, `python -m pyright ...`, `git status`, or `git diff`.
After edits, inspect git diff and run relevant allowed validation commands.
If a tool returns command_denied or validation_denied, do not retry the same
command. Use another allowed validation command or finish with the denial
listed as a validation limitation.
The apply_patch tool input must be an exact patch block starting with
`*** Begin Patch` and ending with `*** End Patch`.
Default to ASCII when creating or editing source files unless non-ASCII content is
explicitly required by the task or already established in the target file.

Final answers must include:
- Change summary
- Modified files
- Validation commands and results
- Remaining risks or follow-up work
Clearly distinguish code failures from environment/tooling blockers.
"""


def extract_patch_block(content: str) -> str | None:
    """Extract a patch block from model output, handling optional code fences."""
    fence_match = re.search(
        r"```(?:diff|patch)?\s*(\*\*\* Begin Patch.*?\*\*\* End Patch)\s*```",
        content,
        re.DOTALL,
    )
    if fence_match:
        return normalize_patch_block(fence_match.group(1))
    patch_match = re.search(r"\*\*\* Begin Patch.*?\*\*\* End Patch", content, re.DOTALL)
    if patch_match:
        return normalize_patch_block(patch_match.group(0))
    return None


def normalize_patch_block(patch: str) -> str:
    """Normalize a patch block to harness format."""
    normalized = patch.strip()
    normalized = re.sub(
        r"^\*\*\* Begin Patch\s+(?=(?:\*\*\* )?(?:Add File|Update File|Delete File): )",
        "*** Begin Patch\n",
        normalized,
    )
    normalized = re.sub(r"\s+\*\*\* End Patch$", "\n*** End Patch", normalized)
    lines = normalized.splitlines()
    cleaned: list[str] = []
    section_mode = ""
    for line in lines:
        stripped = line.strip()
        if not section_mode and re.match(r"^(Add File|Update File|Delete File): ", stripped):
            stripped = f"*** {stripped}"
        if stripped.startswith("*** Add File: "):
            section_mode = "add"
            cleaned.append(stripped.split("```", 1)[0].rstrip())
            continue
        if stripped.startswith("*** Update File: ") or stripped.startswith("*** Delete File: "):
            section_mode = "update" if stripped.startswith("*** Update File: ") else "delete"
            cleaned.append(stripped)
            continue
        if stripped == "*** End Patch":
            section_mode = ""
            cleaned.append(stripped)
            continue
        # Strip model-added Markdown fences around generated Add File content.
        if section_mode == "add" and stripped.startswith("```"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def call_direct_patch_model(
    config: Any,
    trace_writer: Any,
    *,
    task: str,
    task_id: str,
    context: str,
) -> str | None:
    """Call the model for a direct patch edit (no LangChain)."""
    _progress(f"Calling direct patch model for task {task_id}")
    endpoint = config.model.api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model.model_name,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are a coding patch generator.",
                        "Return exactly one patch block and no prose.",
                        "The patch must start with *** Begin Patch and end with *** End Patch.",
                        "Put *** Begin Patch, file section headers, file content, and *** End Patch on separate lines.",
                        "Use only *** Add File:, *** Update File:, or *** Delete File: sections supported by the harness.",
                        "All paths must be relative to the workspace.",
                        "Do not wrap file content in Markdown code fences.",
                        "Use ASCII-only source text unless the task explicitly requires non-ASCII.",
                    ]
                ),
            },
            {"role": "system", "content": f"Repository context:\n{context}"},
            {"role": "user", "content": task},
        ],
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.model.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    trace_writer.append(
        "model_call",
        task_id=task_id,
        input_summary="direct patch edit",
        output_summary="calling OpenAI-compatible chat for patch",
    )
    try:
        with urlopen(request, timeout=config.harness.command_timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        content = str(data["choices"][0]["message"]["content"])
    except (HTTPError, URLError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        _progress(f"Direct patch model failed for task {task_id}: {exc}")
        trace_writer.append(
            "model_fallback",
            task_id=task_id,
            input_summary="direct patch model failed",
            output_summary=str(exc),
            status="failed",
        )
        return None
    _progress(f"Direct patch model returned a response for task {task_id}")
    return extract_patch_block(content)


def call_read_only_model(
    config: Any,
    trace_writer: Any,
    *,
    mode: str,
    task: str,
    task_id: str,
    context: str,
) -> tuple[str | None, str]:
    """Call the model in read-only (ask/plan) mode.

    Returns (response, error_message). On success error_message is empty.
    On failure response is None and error_message describes the issue.
    """
    if not config.model.configured:
        msg = "model is not configured"
        _progress(f"Skipping {mode} model call for task {task_id}: {msg}")
        return None, msg
    if not CHAT_MODEL_AVAILABLE or ChatOpenAI is None or SystemMessage is None or HumanMessage is None:
        _progress(f"Skipping {mode} LangChain model call for task {task_id}: unavailable")
        return call_openai_compatible_chat_model(
            config,
            trace_writer,
            mode=mode,
            task=task,
            task_id=task_id,
            context=context,
        )

    model_options: dict[str, Any] = {}
    if config.model.model_name.startswith("deepseek-v4"):
        model_options["extra_body"] = {"thinking": {"type": "disabled"}}

    model = ChatOpenAI(
        api_key=config.model.api_key,
        base_url=config.model.api_base,
        model=config.model.model_name,
        temperature=0,
        **model_options,
    )
    mode_instruction = (
        "You are in ASK mode. Answer the user's question from the supplied repository "
        "context. Do not propose file modifications unless the user asks for suggestions."
        if mode == "ask"
        else "You are in PLAN mode. Produce a concrete implementation plan with files to "
        "inspect or edit, ordered steps, validation commands, and risks. Do not claim that "
        "files were modified."
    )
    trace_writer.append(
        "model_call",
        task_id=task_id,
        input_summary=f"{mode} read-only context",
        output_summary="calling chat model without tools",
    )
    _progress(f"Calling {mode} read-only model for task {task_id}")
    try:
        response = model.invoke(
            [
                SystemMessage(
                    content="\n".join(
                        [
                            mode_instruction,
                            "You may only use the context already provided.",
                            "No tools are available in this mode.",
                            "Do not say that commands were run or files were modified.",
                        ]
                    )
                ),
                SystemMessage(content=f"Repository context:\n{context}"),
                HumanMessage(content=task),
            ]
        )
    except Exception as exc:  # pragma: no cover - external model fallback
        msg = str(exc)
        _progress(f"{mode} read-only model failed for task {task_id}: {exc}")
        trace_writer.append(
            "model_fallback",
            task_id=task_id,
            input_summary=f"{mode} read-only model failed",
            output_summary=msg,
            status="failed",
        )
        return None, msg
    _progress(f"{mode} read-only model returned a response for task {task_id}")
    return str(getattr(response, "content", response)), ""


def call_openai_compatible_chat_model(
    config: Any,
    trace_writer: Any,
    *,
    mode: str,
    task: str,
    task_id: str,
    context: str,
) -> tuple[str | None, str]:
    """Call an OpenAI-compatible chat endpoint directly (no LangChain).

    Returns (response, error_message). On success error_message is empty.
    On failure response is None and error_message describes the issue.
    """
    mode_instruction = (
        "You are in ASK mode. Answer the user's question from the supplied repository "
        "context. Do not propose file modifications unless the user asks for suggestions."
        if mode == "ask"
        else "You are in PLAN mode. Produce a concrete implementation plan with files to "
        "inspect or edit, ordered steps, validation commands, and risks. Do not claim that "
        "files were modified."
    )
    endpoint = config.model.api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model.model_name,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        mode_instruction,
                        "You may only use the context already provided.",
                        "No tools are available in this mode.",
                        "Do not say that commands were run or files were modified.",
                    ]
                ),
            },
            {"role": "system", "content": f"Repository context:\n{context}"},
            {"role": "user", "content": task},
        ],
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.model.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    trace_writer.append(
        "model_call",
        task_id=task_id,
        input_summary=f"{mode} direct OpenAI-compatible chat",
        output_summary="calling chat model without LangChain",
    )
    _progress(f"Calling {mode} OpenAI-compatible model for task {task_id}")
    try:
        with urlopen(request, timeout=config.harness.command_timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:  # pragma: no cover - external model fallback
        body = exc.read().decode("utf-8", errors="replace")
        msg = f"HTTP {exc.code}: {body}"
        _progress(f"{mode} OpenAI-compatible model failed for task {task_id}: HTTP {exc.code}")
        trace_writer.append(
            "model_fallback",
            task_id=task_id,
            input_summary=f"{mode} direct model failed",
            output_summary=msg,
            status="failed",
        )
        return None, msg
    except URLError as exc:  # pragma: no cover - external model fallback
        msg = f"network error: {exc}"
        _progress(f"{mode} OpenAI-compatible model failed for task {task_id}: {exc}")
        trace_writer.append(
            "model_fallback",
            task_id=task_id,
            input_summary=f"{mode} direct model failed",
            output_summary=msg,
            status="failed",
        )
        return None, msg

    try:
        data = json.loads(raw)
        _progress(f"{mode} OpenAI-compatible model returned a response for task {task_id}")
        return str(data["choices"][0]["message"]["content"]), ""
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        msg = f"unexpected chat response: {exc}"
        _progress(f"{mode} OpenAI-compatible model response could not be parsed for task {task_id}")
        trace_writer.append(
            "model_fallback",
            task_id=task_id,
            input_summary=f"{mode} direct model response parse failed",
            output_summary=msg,
            status="failed",
        )
        return None, msg
