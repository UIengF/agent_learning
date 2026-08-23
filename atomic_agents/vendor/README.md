# Vendored Tools

`vendor/codex/scripts/ask_codex.sh` is the project-local copy internalized from
the global Codex skill. `atomic-agents` uses this copy only, so runtime behavior
is decoupled from `~/.claude`.

The following prerequisites are still external and must be installed on the
machine:

- `codex` CLI available on `PATH`
- `ducc` binary at `~/.comate/baidu-cc/bin/ducc`
- `jq`
- `bash`
- `script`
