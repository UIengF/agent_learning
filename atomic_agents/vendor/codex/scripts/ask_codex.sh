#!/usr/bin/env bash
set -euo pipefail

# --- Timing instrumentation (writes to stderr) ---
if [[ "${CODEX_TIMING:-}" == "1" ]]; then
  _ts() {
    local ts
    ts="$(date +%s%3N 2>/dev/null || true)"
    if [[ "$ts" =~ ^[0-9]+$ ]]; then
      echo "$ts"
    else
      python -c "import time; print(int(time.time()*1000))"
    fi
  }
  _timing_events_file=""
  _timing_record() {
    local key="$1" ts
    ts="$(_ts)"
    echo "[timing] ${key}=${ts}" >&2
    if [[ -n "${_timing_events_file:-}" ]]; then
      printf '%s=%s\n' "$key" "$ts" >> "$_timing_events_file"
    fi
  }
  _timing_first() {
    local key="$1"
    [[ -n "${_timing_events_file:-}" && -s "$_timing_events_file" ]] || return 0
    awk -F= -v key="$key" '$1 == key { print $2; exit }' "$_timing_events_file"
  }
  _timing_value() {
    local key="$1" value="$2"
    if [[ -n "$value" ]]; then
      echo "[timing] ${key}=${value}" >&2
    else
      echo "[timing] ${key}=NA" >&2
    fi
  }
  _timing_delta() {
    local label="$1" start="$2" finish="$3"
    if [[ "$start" =~ ^[0-9]+$ && "$finish" =~ ^[0-9]+$ ]]; then
      echo "[timing] ${label}=$((finish - start))ms" >&2
    else
      echo "[timing] ${label}=NA" >&2
    fi
  }
  _t_start=$(_ts)
  echo "[timing] script_start=$_t_start" >&2
fi

usage() {
  cat <<'USAGE'
Usage:
  ask_codex.sh <task> [options]
  ask_codex.sh -t <task> [options]

Task input:
  <task>                       First positional argument is the task text
  -t, --task <text>            Alias for positional task (backward compat)
  (stdin)                      Pipe task text via stdin if no arg/flag given

File context (optional, repeatable):
  -f, --file <path>            Priority file path

Multi-turn:
      --session <id>           Resume a previous session (thread_id from prior run)

Options:
  -w, --workspace <path>       Workspace directory (default: current directory)
      --model <name>           Model override
      --reasoning <level>      Reasoning effort: none, minimal, low, medium, high, xhigh (default: xhigh)
      --service-tier <tier>    Service tier / speed: auto, default, flex, priority, scale
      --sandbox <mode>         Sandbox mode override
      --read-only              Read-only sandbox (no file changes)
      --full-auto              Full-auto mode (default)
  -o, --output <path>          Output file path
  -h, --help                   Show this help

Output (on success):
  session_id=<thread_id>       Use with --session for follow-up calls
  output_path=<file>           Path to response markdown

Examples:
  # New task (positional)
  ask_codex.sh "Add error handling to api.ts" -f src/api.ts

  # With explicit workspace
  ask_codex.sh "Fix the bug" -w /other/repo

  # Continue conversation
  ask_codex.sh "Also add retry logic" --session <id>
USAGE
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing required command: $1" >&2
    exit 1
  fi
}

trim_whitespace() {
  awk 'BEGIN { RS=""; ORS="" } { gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, ""); print }' <<<"$1"
}

to_abs_if_exists() {
  local target="$1"
  if [[ -e "$target" ]]; then
    local dir
    dir="$(cd "$(dirname "$target")" && pwd)"
    echo "$dir/$(basename "$target")"
    return
  fi
  echo "$target"
}

resolve_file_ref() {
  local workspace="$1" raw="$2" cleaned
  cleaned="$(trim_whitespace "$raw")"
  [[ -z "$cleaned" ]] && { echo ""; return; }
  if [[ "$cleaned" =~ ^(.+)#L[0-9]+$ ]]; then cleaned="${BASH_REMATCH[1]}"; fi
  if [[ "$cleaned" =~ ^(.+):[0-9]+(-[0-9]+)?$ ]]; then cleaned="${BASH_REMATCH[1]}"; fi
  if [[ "$cleaned" != /* ]]; then cleaned="$workspace/$cleaned"; fi
  to_abs_if_exists "$cleaned"
}

append_file_refs() {
  local raw="$1" item
  IFS=',' read -r -a items <<< "$raw"
  for item in "${items[@]}"; do
    local trimmed
    trimmed="$(trim_whitespace "$item")"
    [[ -n "$trimmed" ]] && file_refs+=("$trimmed")
  done
}

# --- Parse arguments ---

workspace="${PWD}"
task_text=""
model=""
reasoning_effort=""
service_tier=""
sandbox_mode=""
read_only=false
full_auto=true
output_path=""
session_id=""
file_refs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workspace)   workspace="${2:-}"; shift 2 ;;
    -t|--task)        task_text="${2:-}"; shift 2 ;;
    -f|--file|--focus) append_file_refs "${2:-}"; shift 2 ;;
    --model)          model="${2:-}"; shift 2 ;;
    --reasoning)      reasoning_effort="${2:-}"; shift 2 ;;
    --service-tier)   service_tier="${2:-}"; shift 2 ;;
    --sandbox)        sandbox_mode="${2:-}"; full_auto=false; shift 2 ;;
    --read-only)      read_only=true; full_auto=false; shift ;;
    --full-auto)      full_auto=true; shift ;;
    --session)        session_id="${2:-}"; shift 2 ;;
    -o|--output)      output_path="${2:-}"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    -*)               echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)                if [[ -z "$task_text" ]]; then task_text="$1"; shift; else echo "[ERROR] Unexpected argument: $1" >&2; usage >&2; exit 1; fi ;;
  esac
done

require_cmd codex
require_cmd jq

# --- Validate inputs ---

if [[ ! -d "$workspace" ]]; then
  echo "[ERROR] Workspace does not exist: $workspace" >&2; exit 1
fi
workspace="$(cd "$workspace" && pwd)"

if [[ -z "$task_text" && ! -t 0 ]]; then
  task_text="$(cat)"
fi
task_text="$(trim_whitespace "$task_text")"

if [[ -z "$task_text" ]]; then
  echo "[ERROR] Request text is empty. Pass a positional arg, --task, or stdin." >&2; exit 1
fi

# --- Prepare output path ---

if [[ -z "$output_path" ]]; then
  timestamp="$(date -u +"%Y%m%d-%H%M%S")"
  skill_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  output_path="$skill_dir/.runtime/${timestamp}.md"
fi
output_dir="$(dirname "$output_path")"
[[ -d "$output_dir" ]] || mkdir -p "$output_dir"
output_base="$(basename "$output_path")"
raw_events_path="$output_dir/${output_base%.*}.events.jsonl"

# --- Build file context block ---

file_block=""
if (( ${#file_refs[@]} > 0 )); then
  file_block=$'\nPriority files (read these first before making changes):'
  for ref in "${file_refs[@]}"; do
    resolved="$(resolve_file_ref "$workspace" "$ref")"
    [[ -z "$resolved" ]] && continue
    exists_tag="missing"
    [[ -e "$resolved" ]] && exists_tag="exists"
    file_block+=$'\n- '"${resolved} (${exists_tag})"
  done
fi

# --- Build prompt ---

prompt="$task_text"
if [[ -n "$file_block" ]]; then
  prompt+=$'\n'"$file_block"
fi

# --- Determine reasoning effort ---

if [[ -z "$reasoning_effort" ]]; then
  reasoning_effort="xhigh"
fi

# --- Build codex command ---

if [[ -n "$session_id" ]]; then
  # Resume mode: continue a previous session.
  # Note: --sandbox/--full-auto are omitted because `codex exec resume`
  # reuses the sandbox settings from the original session.
  cmd=(codex exec resume --skip-git-repo-check --json -c "model_reasoning_effort=\"$reasoning_effort\"")
  [[ -n "$service_tier" ]] && cmd+=(-c "service_tier=\"$service_tier\"")
  [[ -n "$model" ]] && cmd+=(-m "$model")
  cmd+=("$session_id")
else
  # New session
  cmd=(codex exec --cd "$workspace" --skip-git-repo-check --json -c "model_reasoning_effort=\"$reasoning_effort\"")
  [[ -n "$service_tier" ]] && cmd+=(-c "service_tier=\"$service_tier\"")
  if [[ "$read_only" == true ]]; then
    cmd+=(--sandbox read-only)
  elif [[ -n "$sandbox_mode" ]]; then
    cmd+=(--sandbox "$sandbox_mode")
  elif [[ "$full_auto" == true ]]; then
    # Default: bypass approvals + sandbox so codex's own cleanup commands
    # (e.g. `rm` of generated .pyc) are not rejected mid-turn — which otherwise
    # surfaces as a turn.failed/stream-disconnect even though the real work
    # already completed. EXTREMELY DANGEROUS: codex runs commands unsandboxed.
    cmd+=(--dangerously-bypass-approvals-and-sandbox)
  fi
  [[ -n "$model" ]] && cmd+=(-m "$model")
fi

# --- Progress watcher function ---

print_progress() {
  local line="$1"
  local item_type cmd_str preview
  # Fast string checks before calling jq
  case "$line" in
    *'"item.started"'*'"command_execution"'*)
      cmd_str=$(printf '%s' "$line" | jq -r '.item.command // empty' 2>/dev/null | sed 's|^/bin/zsh -lc ||; s|^/bin/bash -c ||' | cut -c1-100)
      [[ -n "$cmd_str" ]] && echo "[codex] > $cmd_str" >&2
      ;;
    *'"item.completed"'*'"agent_message"'*)
      preview=$(printf '%s' "$line" | jq -r '.item.text // empty' 2>/dev/null | head -1 | cut -c1-120)
      [[ -n "$preview" ]] && echo "[codex] $preview" >&2
      ;;
  esac
}

# --- Execute and capture JSON output ---

stderr_file="$(mktemp)"
json_file="$(mktemp)"
prompt_file="$(mktemp)"
timing_events_file="$(mktemp)"
trap 'rm -f "$stderr_file" "$json_file" "$prompt_file" "$timing_events_file"' EXIT
if [[ "${CODEX_TIMING:-}" == "1" ]]; then
  _timing_events_file="$timing_events_file"
fi

# Surface the live event file path on stderr *before* the child starts, so a
# caller (e.g. the Python adapter's idle-timeout watcher) can tail json_file
# for real-time activity while codex is still running — json_file is written
# to incrementally by process_output() below (line-buffered via `script`),
# unlike raw_events_path, which is only cp'd into place after the run ends.
echo "[codex] live_events_path=$json_file" >&2

# Write prompt to a temp file and pipe from there to avoid shell argument
# length issues and encoding problems with very long or multi-byte prompts.
printf "%s" "$prompt" > "$prompt_file"

# Use a pseudo-TTY to force line-buffered JSONL output from codex on Linux/macOS
# (via `script`). On Windows, `script` is not available so we fall back to direct
# execution — output may block-buffer and progress won't stream in real-time, but
# functional results are identical.

inner_cmd="cd $(printf '%q' "$workspace") && $(printf '%q ' "${cmd[@]}") < $(printf '%q' "$prompt_file") 2>$(printf '%q' "$stderr_file")"

process_output() {
  while IFS= read -r line; do
    if [[ "${CODEX_TIMING:-}" == "1" ]]; then
      case "$line" in
        *'"thread.started"'*)  _timing_record "event_thread_started" ;;
        *'"turn.started"'*)    _timing_record "event_turn_started" ;;
        *'"agent_message"'*)   _timing_record "event_agent_message" ;;
        *'"turn.completed"'*)  _timing_record "event_turn_completed" ;;
      esac
    fi
    # Strip terminal artifacts (carriage return, ^D EOF marker, ANSI escapes, other control chars)
    cleaned="${line//$'\r'/}"
    cleaned="${cleaned//$'\004'/}"
    # Strip ANSI escape sequences: remove the ESC byte first, then CSI sequences, then non-JSON prefix
    cleaned="$(printf '%s' "$cleaned" | tr -d '\033' | sed 's/\[[0-9;]*[a-zA-Z]//g; s/^[^{]*//')"
    [[ -z "$cleaned" ]] && continue
    # Only process JSON lines (must start with '{')
    [[ "$cleaned" != \{* ]] && continue
    # Write to json_file for later parsing
    printf '%s\n' "$cleaned" >> "$json_file"
    # Only parse progress-relevant events (fast string check before jq)
    case "$cleaned" in
      *'"item.started"'*|*'"item.completed"'*) print_progress "$cleaned" ;;
    esac
  done
}

pipeline_exit=0
if [[ "${CODEX_TIMING:-}" == "1" ]]; then
  _t_before_exec=$(_ts)
  echo "[timing] before_codex_exec=$_t_before_exec" >&2
fi
if command -v script &>/dev/null; then
  # Linux/macOS: script creates a pseudo-TTY so codex line-buffers its JSONL output
  script -q /dev/null /bin/bash -c "$inner_cmd" | process_output || pipeline_exit=$?
else
  # Windows / fallback: run directly via bash -c (same command string as the
  # script path). No pseudo-TTY available, so output may block-buffer and
  # progress won't stream in real-time, but results are still correct.
  /bin/bash -c "$inner_cmd" | process_output || pipeline_exit=$?
fi

if [[ -s "$stderr_file" ]] && grep -q '\[ERROR\]' "$stderr_file" 2>/dev/null; then
  echo "[ERROR] Codex command failed" >&2
  cat "$stderr_file" >&2
  exit 1
fi

if [[ -s "$stderr_file" ]]; then
  cat "$stderr_file" >&2
fi

# Detect Codex execution failure from the JSON event stream. The `script`
# pseudo-TTY wrapper masks the inner command's real exit code, so pipeline_exit
# alone is not trustworthy. Codex emits explicit error/turn.failed events when a
# turn fails (e.g. invalid reasoning effort, API 4xx); surface those as a hard
# failure so codex_workflow.py doesn't misclassify a real failure as a
# "missing handoff" soft-degrade.
#
# IMPORTANT — judge by TERMINAL state, not by any error event appearing:
# Codex's SSE stream can emit a recoverable `{"type":"error","message":
# "Reconnecting... N/5 (stream disconnected ...)"}` mid-turn and then
# auto-reconnect and finish normally (ending with `turn.completed`). Treating
# any `type==error` as fatal kills these self-healed runs. So:
#   - `turn.failed`                  -> hard failure (always)
#   - a `turn.completed` exists      -> success, even if reconnect errors occurred
#   - `type==error` with NO terminal `turn.completed` -> real failure (stream
#     dropped and never recovered)
# Use jq for structured detection (only the top-level .type field counts) so a
# nested/escaped "type":"error" inside an agent_message body cannot trigger a
# false positive. Fall back to a line grep only if jq somehow can't parse.
codex_failed=false
fail_reason=""
if [[ -s "$json_file" ]]; then
  if jq -e 'select(.type == "turn.failed")' "$json_file" >/dev/null 2>&1; then
    codex_failed=true
    fail_reason="turn.failed"
  elif jq -e 'select(.type == "turn.completed")' "$json_file" >/dev/null 2>&1; then
    # Turn completed — any earlier error events were recoverable reconnects.
    codex_failed=false
    if jq -e 'select(.type == "error")' "$json_file" >/dev/null 2>&1; then
      echo "[codex] note: recovered from a transient stream error (reconnect), turn completed OK" >&2
    fi
  elif jq -e 'select(.type == "error")' "$json_file" >/dev/null 2>&1; then
    # Error event(s) but no terminal turn.completed: stream dropped unrecovered.
    codex_failed=true
    fail_reason="error (no turn.completed — stream did not recover)"
  fi
fi
if [[ "$codex_failed" == true ]]; then
  echo "[ERROR] Codex reported a fatal event: ${fail_reason}" >&2
  jq -c 'select(.type == "turn.failed" or .type == "error")' "$json_file" 2>/dev/null | head -3 >&2
  # Still preserve the raw event stream for debugging before exiting.
  cp "$json_file" "$raw_events_path" 2>/dev/null || true
  exit 1
fi
if [[ "${pipeline_exit:-0}" -ne 0 ]]; then
  echo "[ERROR] Codex pipeline exited with code $pipeline_exit" >&2
  cp "$json_file" "$raw_events_path" 2>/dev/null || true
  exit "$pipeline_exit"
fi

# Preserve the complete raw Codex JSON event stream next to the markdown
# summary so downstream timeline tools can inspect full-fidelity events.
cp "$json_file" "$raw_events_path"

# --- Extract thread_id and all messages from JSON stream ---

thread_id="$(jq -r 'select(.type == "thread.started") | .thread_id' < "$json_file" | head -1)"

# Collect agent messages only. The complete raw event stream remains available
# at raw_events_path for tooling that needs command or file-operation details.
jq -r '
  select(.type == "item.completed" and .item.type == "agent_message")
  | .item.text
' < "$json_file" 2>/dev/null > "$output_path"

# If nothing was captured, write a fallback
if [[ ! -s "$output_path" ]]; then
  echo "(no response from codex)" > "$output_path"
fi

# --- Output results ---

if [[ "${CODEX_TIMING:-}" == "1" ]]; then
  _t_done=$(_ts)
  _t_event_thread_started="$(_timing_first "event_thread_started")"
  _t_event_turn_started="$(_timing_first "event_turn_started")"
  _t_event_agent_message="$(_timing_first "event_agent_message")"
  _t_event_turn_completed="$(_timing_first "event_turn_completed")"

  echo "[timing] summary_start" >&2
  _timing_value "summary.script_start" "${_t_start:-}"
  _timing_value "summary.before_codex_exec" "${_t_before_exec:-}"
  _timing_value "summary.event_thread_started" "$_t_event_thread_started"
  _timing_value "summary.event_turn_started" "$_t_event_turn_started"
  _timing_value "summary.event_agent_message" "$_t_event_agent_message"
  _timing_value "summary.event_turn_completed" "$_t_event_turn_completed"
  _timing_value "summary.script_done" "$_t_done"
  _timing_delta "delta.script_start_to_before_codex_exec" "${_t_start:-}" "${_t_before_exec:-}"
  _timing_delta "delta.before_codex_exec_to_event_thread_started" "${_t_before_exec:-}" "$_t_event_thread_started"
  _timing_delta "delta.event_thread_started_to_event_turn_started" "$_t_event_thread_started" "$_t_event_turn_started"
  _timing_delta "delta.event_turn_started_to_event_agent_message" "$_t_event_turn_started" "$_t_event_agent_message"
  _timing_delta "delta.event_agent_message_to_event_turn_completed" "$_t_event_agent_message" "$_t_event_turn_completed"
  _timing_delta "delta.event_turn_completed_to_script_done" "$_t_event_turn_completed" "$_t_done"
  _timing_delta "delta.before_codex_exec_to_script_done" "${_t_before_exec:-}" "$_t_done"
  _timing_delta "delta.script_start_to_script_done" "${_t_start:-}" "$_t_done"
  echo "[timing] summary_end" >&2
fi

if [[ -n "$thread_id" ]]; then
  echo "session_id=$thread_id"
fi
echo "output_path=$output_path"
echo "raw_events_path=$raw_events_path"
