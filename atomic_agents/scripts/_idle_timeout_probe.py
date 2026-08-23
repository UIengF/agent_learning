"""Minimal empirical test of idle-timeout watch behavior.

Directly exercises run_with_timeout from the adapters module with a tiny
timeout_sec, to prove whether periodically touching a file in watch_dir
refreshes the idle timer (Direction 2 feasibility).
"""
import sys, os, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import atomic_agents.adapters as ad

TIMEOUT = 6  # seconds idle budget

def run_case(name, script):
    wd = tempfile.mkdtemp(prefix=f"idletest-{name}-")
    t0 = time.monotonic()
    timed_out = False
    rc = None
    try:
        cp = ad.run_with_timeout(
            ["/bin/bash", "-c", script],
            TIMEOUT,
            cwd=wd,
            stdin_text=None,
            watch_dir=wd,
        )
        rc = cp.returncode
    except ad.AdapterTimeout as e:
        timed_out = True
    dur = time.monotonic() - t0
    print(f"[{name}] timed_out={timed_out} returncode={rc} dur={dur:.1f}s  watch={wd}")
    return timed_out, rc, dur

if __name__ == "__main__":
    print(f"idle TIMEOUT={TIMEOUT}s\n")
    # Case A: total runtime > timeout, but NEVER writes to watch_dir -> expect KILLED (timed_out)
    run_case("idle-silent", "sleep 10; echo done")
    print()
    # Case B: same total runtime, but touches a file every 2s (< timeout) -> expect SURVIVES
    run_case("touch-loop", "for i in 1 2 3 4 5; do echo line$i >> out.md; sleep 2; done; echo done")
