#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASK_CODEX="$ROOT_DIR/vendor/codex/scripts/ask_codex.sh"
DUCC_BIN="$HOME/.comate/baidu-cc/bin/ducc"

missing=()

if [[ ! -f "$ASK_CODEX" ]]; then
  missing+=("vendor/codex/scripts/ask_codex.sh is missing")
elif [[ ! -x "$ASK_CODEX" ]]; then
  missing+=("vendor/codex/scripts/ask_codex.sh is not executable")
fi

if ! command -v codex >/dev/null 2>&1; then
  missing+=("codex command is not on PATH")
fi

if [[ ! -x "$DUCC_BIN" ]]; then
  missing+=("ducc binary is missing or not executable: $DUCC_BIN")
fi

if ! command -v jq >/dev/null 2>&1; then
  missing+=("jq command is not on PATH")
fi

if (( ${#missing[@]} > 0 )); then
  echo "Dependency check failed:"
  for item in "${missing[@]}"; do
    echo "- $item"
  done
  exit 1
fi

echo "Dependency check passed:"
echo "- vendor/codex/scripts/ask_codex.sh exists and is executable"
echo "- codex command is on PATH: $(command -v codex)"
echo "- ducc binary exists: $DUCC_BIN"
echo "- jq command is on PATH: $(command -v jq)"
