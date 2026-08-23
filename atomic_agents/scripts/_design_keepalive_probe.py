"""Short probe: does a single REAL codex design node actually write _progress.md?

Builds one AtomContract for a design node (read_only=False, like write_node
produces), invokes the codex adapter directly, and a background watcher thread
records whether _progress.md appears and how its mtime advances during the run.

Goal: in ~10 min decide if the prompt keepalive works BEFORE committing to a
full multi-node orchestration. Uses a SHORTENED idle timeout so a silent codex
dies fast instead of burning 1800s.
"""
import sys, os, time, threading, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import atomic_agents.adapters as ad
from atomic_agents.adapters.codex import CodexAdapter
from atomic_agents.scheduler import build_contract
from orchestrations.common import make_skeleton, write_node
from orchestrations import prompts

WS = os.path.join(os.path.dirname(__file__), "..", "atomic-orch-out", "carimg-probe-design")
WS = os.path.abspath(WS)
OUTPUT_FILE = "design-stance0-v1.md"
PROGRESS = "_progress.md"
IDLE_TIMEOUT = 420  # 7 min idle budget for the probe (not the prod 1800)

stance = {"stance_name": "保守兼容派", "focus": "复用现有、最小改动、控制迁移风险",
          "prompt_hint": "优先从复用现有 schema、最小改动角度设计。"}

def build_request():
    # Reuse the real research report + a trimmed instruction so codex has real work.
    req_path = os.path.join(WS, "request.txt")
    with open(req_path, encoding="utf-8") as f:
        return f.read()

def watcher(stop_evt, log):
    progress_path = os.path.join(WS, PROGRESS)
    appeared_at = None
    last_mtime = None
    samples = 0
    while not stop_evt.is_set():
        if os.path.exists(progress_path):
            m = os.stat(progress_path).st_mtime
            if appeared_at is None:
                appeared_at = time.time()
                log.append(f"[watch] _progress.md APPEARED at +{appeared_at-START:.0f}s size={os.path.getsize(progress_path)}")
            if last_mtime is None or m > last_mtime:
                last_mtime = m
                samples += 1
                log.append(f"[watch] _progress.md mtime advanced (#{samples}) +{time.time()-START:.0f}s size={os.path.getsize(progress_path)}")
        time.sleep(5)
    log.append(f"[watch] done: appeared={appeared_at is not None} mtime_advances={samples}")

if __name__ == "__main__":
    os.makedirs(WS, exist_ok=True)
    # copy the real research report + request into the probe workspace
    src_ws = os.path.join(os.path.dirname(__file__), "..", "atomic-orch-out", "car-image-evidence-v2")
    src_ws = os.path.abspath(src_ws)
    shutil.copy(os.path.join(src_ws, "research-2.md"), os.path.join(WS, "research.md"))
    shutil.copy(os.path.join(src_ws, "request.txt"), os.path.join(WS, "request.txt"))

    task = prompts.design_task(build_request(), stance, "research.md", 1)
    node = write_node(node_id="design_0_v1_probe", role="designer",
                      task=task, output_file=OUTPUT_FILE)
    skeleton = make_skeleton("probe", [node])
    contract = build_contract(node, skeleton, {}, "probe", workspace=WS)

    START = time.time()
    log = []
    stop = threading.Event()
    wt = threading.Thread(target=watcher, args=(stop, log), daemon=True)
    wt.start()
    print(f"[probe] invoking REAL codex design node, idle_timeout={IDLE_TIMEOUT}s, ws={WS}", flush=True)
    adapter = CodexAdapter()
    res = adapter.invoke(contract, timeout_sec=IDLE_TIMEOUT)
    stop.set(); wt.join(timeout=8)
    dur = time.time() - START
    print("\n".join(log))
    print(f"\n[probe] RESULT status={res.status} dur={dur:.0f}s")
    if res.error:
        print(f"[probe] error: {res.error[:400]}")
    out_path = os.path.join(WS, OUTPUT_FILE)
    print(f"[probe] final output {OUTPUT_FILE} exists={os.path.exists(out_path)} "
          f"size={os.path.getsize(out_path) if os.path.exists(out_path) else 0}")
    prog_path = os.path.join(WS, PROGRESS)
    print(f"[probe] _progress.md exists={os.path.exists(prog_path)} "
          f"size={os.path.getsize(prog_path) if os.path.exists(prog_path) else 0}")
