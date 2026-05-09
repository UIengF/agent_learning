# Eval Boundary Suite

`boundary_suite.json` is the first batch of 12 eval cases for widening the agent task boundary.

The cases intentionally cover:

- hidden validation pass/fail/denied behavior
- two multi-turn `group_id` / `continue_session` workflows
- `setup_files` seeded workspaces
- BOM-safe CSV CLI work
- JSON transformation
- hidden edge cases beyond visible tests
- refusal/recovery around unsafe command requests
- ambiguous structured output

Hidden checks live in `hidden/boundary_checks.py`. The suite uses `{{suite_dir}}` placeholders, which `load_eval_suite` expands to the suite file directory before execution. The expanded hidden validation commands are never appended to the agent prompt.
