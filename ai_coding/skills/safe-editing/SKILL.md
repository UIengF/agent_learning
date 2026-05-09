---
name: safe-editing
description: Apply targeted code changes with a plan, scoped patch, diff review, and validation.
---

Use this skill for implementation tasks that modify files.

Rules:

1. Call `plan_update` before `apply_patch`.
2. Keep patches narrow and workspace-scoped.
3. Inspect `git diff` after editing.
4. Run the smallest relevant allowed validation command.
5. Report changed files, validation result, and residual risk.
