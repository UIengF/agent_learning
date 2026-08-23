# Solution Design Example

This example builds a solution-design orchestration on top of `atomic-agents`.
It turns one design request into a fixed DAG after a short preflight phase:

```text
Phase 0: research.md
  -> Phase 1: select N stances
  -> Phase 2:
       N parallel v1 designs
       -> red-team round 1
       -> N parallel v2 designs
       -> red-team round 2
       -> N parallel v3 designs
       -> final synthesis
```

Red-team and synthesis nodes are ordinary writer atoms. They do not use the
special `reviewer` role, so the scheduler will not interpret their output as
pass/fail verdict JSON.

## Run

From the repository root:

```bash
.venv/bin/python3 examples/solution_design.py "你的设计需求"
```

With no argument it runs a demo request:

```text
设计一个 Python 进程内的 LRU 缓存，支持 TTL 过期和线程安全
```

The script writes output under:

```text
examples/solution-design-out/
```

## Outputs

- `research.md`: web-backed research report from Phase 0.
- `design-stance{i}-v1.md`: first design from stance `i`.
- `challenge-r1.md`: first red-team challenge report.
- `design-stance{i}-v2.md`: revised design after round 1.
- `challenge-r2.md`: second red-team challenge report.
- `design-stance{i}-v3.md`: final design from stance `i`.
- `final-solution.md`: synthesized final answer.

Phase 2 lock events are available from `result.run_result.lock_dir`. The
example keeps locks outside the workspace so scheduler drift detection only
sees intended design artifacts.

## Reused Atomic-Agents Capabilities

- `run_skeleton`: executes the static Phase 2 DAG.
- Disjoint `write_scope`: lets same-version stance designers run in parallel.
- `output_file` flow: downstream nodes receive upstream files through
  `inputs=[{"from": node_id, "field": "output_file"}]`.
- Real adapters: defaults to `DuccAdapter`, and the `RoleRoutingAdapter` can
  route `designer`, `redteam`, and `synthesizer` roles to different adapters
  such as `CodexAdapter` or `DuccAdapter`.

## Boundaries

- Phase 1 chooses stances before `run_skeleton`; the DAG is static during the
  run.
- Red-team depth is fixed at two rounds in this example. The `rounds` parameter
  is reserved and currently only accepts `2`.
- `research.md` is produced before the skeleton run, so design tasks refer to it
  by filename in the shared workspace rather than through an upstream skeleton
  input.
