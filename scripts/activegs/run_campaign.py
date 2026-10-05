"""Linux-only, stdlib-only orchestration of the frozen optimization-v1 campaign.

Plan/status do not inspect GPUs, create files, or import reconstruction libraries.
Tick advances at most one stage; a scheduler may call it repeatedly. It never
kills a process, retries failed stages, or replaces existing experiment outputs.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from viewmend3d import launch_coordination as coordination
from viewmend3d.launch_coordination import launch_lock, blocking_web_job

VERSION = "viewmend-campaign-v1"
NAME = "optimization-v1"
MAIN_METHODS = ["confidence_nooracle", "random_matched", "defect"]
STAGES = [
    {"name": "smoke", "methods": ["confidence_nooracle", "defect", "refine_only"],
     "seeds": [0], "frames": 6, "prefix": 2, "mode": "observations", "seconds": 180},
    {"name": "observations", "methods": MAIN_METHODS + ["defect_no_gate", "refine_only"],
     "seeds": [0, 1, 2], "frames": 60, "prefix": 20, "mode": "observations", "seconds": 180},
    {"name": "time", "methods": MAIN_METHODS, "seeds": [0, 1, 2],
     "frames": 60, "prefix": 20, "mode": "time", "seconds": 180},
]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def location(root):
    return Path(root) / "runs" / "campaigns" / NAME


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_state(root):
    path = location(root) / "state.json"
    if not path.exists():
        return {"version": VERSION, "status": "not_started", "stage_index": 0,
                "source_identity": None, "attempts": [], "active": None}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != VERSION or type(state.get("stage_index")) is not int:
        raise ValueError("Unsupported or malformed campaign state; inspect it manually")
    if not 0 <= state["stage_index"] <= len(STAGES):
        raise ValueError("Invalid campaign stage index")
    return state


@contextmanager
def campaign_lock(root):
    # Imported only for mutations, so plan/status and mocked CPU tests are portable.
    if sys.platform != "linux":
        raise RuntimeError("Campaign execution is supported only on the Linux GPU server")
    import fcntl
    directory = location(root)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "state.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def save_state(root, state):
    state["updated_at"] = timestamp()
    active = state.get("active")
    if active:
        for index, attempt in enumerate(state["attempts"]):
            if attempt["token"] == active["token"]:
                state["attempts"][index] = dict(active)
    atomic_json(location(root) / "state.json", state)


proc_start_identity = coordination.proc_start_identity
proc_identity = coordination.proc_identity


def process_alive(identity):
    return coordination.process_alive(identity, inspect=proc_identity)


def observe_started_process(process, argv, timeout=1.0):
    return coordination.observe_started_process(process, argv, timeout,
                                                inspect=proc_identity, inspect_start=proc_start_identity)


def find_worker(argv):
    """Recover the narrow launch-before-PID-save crash window, without guessing."""
    matches = []
    for candidate in Path("/proc").iterdir():
        if candidate.name.isdigit():
            identity = proc_identity(int(candidate.name))
            if identity and identity["argv"] == argv:
                matches.append(identity)
    if len(matches) > 1:
        raise RuntimeError("Duplicate managed workers found; manual inspection required")
    return matches[0] if matches else None


def read_gpus():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, check=True, timeout=10)
    rows = []
    for line in result.stdout.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        rows.append({"index": index, "memory_mb": memory, "utilization": utilization})
    if not rows:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return rows


def idle_gpus(rows):
    return [row for row in rows if row["memory_mb"] < 1024 and row["utilization"] <= 5]


def source_identity(root):
    """Freeze project/upstream commits and actual Python/shell/config contents."""
    def commit(directory):
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=directory,
                              capture_output=True, text=True, check=True).stdout.strip()
    def digest(directory, subdirectories):
        files = sorted({path for subdirectory in subdirectories
                        for path in (directory / subdirectory).rglob("*")
                        if path.is_file() and path.suffix in (".py", ".sh", ".yaml", ".yml")})
        if not files:
            raise ValueError(f"No source files found in {directory}")
        result = hashlib.sha256()
        for path in files:
            result.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
            result.update(hashlib.sha256(path.read_bytes()).digest())
        return result.hexdigest()
    root = Path(root)
    upstream = root / "external" / "active-gs"
    if not (upstream / "config/main.yaml").is_file():
        raise ValueError("Pinned ActiveGS checkout is missing")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True, check=True).stdout
    if dirty:
        raise RuntimeError("Main project working tree is dirty; commit and review before campaign execution")
    # Deployment preflight permits only the documented multiprocessing patch.
    # Freeze that actual diff as well as all runtime files for the entire chain.
    upstream_diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=upstream,
                                   capture_output=True, check=True).stdout
    return {"project_commit": commit(root), "project_source_sha256": digest(root, ("src", "scripts")),
            "upstream_commit": commit(upstream), "upstream_source_sha256": digest(upstream, (".",)),
            "upstream_diff_sha256": hashlib.sha256(upstream_diff).hexdigest()}


def benchmark_argv(root, stage, attempt):
    python = Path(root) / ".envs/activegs/bin/python"
    return [str(python), str(Path(root) / "scripts/activegs/run_benchmark.py"),
            "--upstream", str(Path(root) / "external/active-gs"),
            "--run-dir", attempt["run_dir"], "--gpu", str(attempt["gpu"]),
            "--scene", "replica/office0", "--methods", *stage["methods"],
            "--seeds", *map(str, stage["seeds"]), "--frames", str(stage["frames"]),
            "--prefix-frames", str(stage["prefix"]), "--protocol", stage["mode"],
            "--budget", str(stage["seconds"])]


def experiment_environment(root, gpu):
    root = Path(root)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               MPLBACKEND="Agg", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               TORCH_HOME=str(root / "runs/torch-cache"))
    directories = [root / ".sysroot/usr/lib/x86_64-linux-gnu", root / ".envs/activegs/lib"]
    env["LD_LIBRARY_PATH"] = ":".join(map(str, directories)) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def failed(root, state, error):
    state.update(status="failed", error=str(error))
    state.pop("waiting_reason", None)
    state.pop("blocking_job", None)
    if (state.get("active") or {}).get("status") in ("starting", "running"):
        state["active"].update(status="failed", error=str(error), finished_at=timestamp())
    save_state(root, state)
    return state


def spawn_worker(argv, log):
    with Path(log).open("ab", buffering=0) as stream:
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                close_fds=True)


def tick(root):
    # Every launcher acquires this outer lock before its own job/state lock.
    # Campaign workers only use state locks; web workers never take this
    # campaign lock, so they cannot form the opposite lock order.
    root = Path(root).resolve()
    with launch_lock(root):
        return _tick(root)


def _tick(root):
    root = Path(root).resolve()
    with campaign_lock(root):
        state = read_state(root)
        active = state.get("active")
        if active:
            # Check a still-live worker/child before considering any terminal file.
            for kind in ("worker", "child"):
                if process_alive(active.get(kind)):
                    return {**state, "live_process": kind}
                start = active.get(f"{kind}_start")
                if start and proc_start_identity(start["pid"]) == start:
                    observed = proc_identity(start["pid"])
                    if observed and observed["argv"] == active[f"{kind}_argv"]:
                        active[kind] = observed
                        active.pop(f"{kind}_start")
                        if kind == "worker":
                            active["status"] = "running"
                        save_state(root, state)
                    return {**state, "live_process": f"{kind}_starting"}
            if active.get("status") == "starting":
                recovered = find_worker(active["worker_argv"])
                if recovered:
                    active["worker"] = recovered
                    active["status"] = "running"
                    state["status"] = "running"
                    save_state(root, state)
                    return state
            if active.get("status") not in ("completed", "deferred", "failed"):
                return failed(root, state, "Managed worker/child exited without verified completion; no automatic retry")
        if state["status"] in ("failed", "completed"):
            return state
        if state["stage_index"] == len(STAGES):
            state["status"] = "completed"
            state.pop("waiting_reason", None)
            state.pop("blocking_job", None)
            save_state(root, state)
            return state
        try:
            blocker = blocking_web_job(root)
            if blocker:
                state.update(status="waiting", waiting_reason="managed_web_job", blocking_job=blocker)
                save_state(root, state)
                return state
            rows = read_gpus()
            available = idle_gpus(rows)
            state["gpu_check"] = {"checked_at": timestamp(), "devices": rows}
            if not available:
                state.update(status="waiting", waiting_reason="gpu_busy")
                state.pop("blocking_job", None)
                save_state(root, state)
                return state
            identity = source_identity(root)
            if state["source_identity"] is not None and identity != state["source_identity"]:
                raise RuntimeError("Frozen campaign source changed; inspect and create a new reviewed campaign")
            state["source_identity"] = identity
            stage = STAGES[state["stage_index"]]
            token = uuid.uuid4().hex
            directory = location(root)
            (directory / "logs").mkdir(exist_ok=True)
            (directory / "stages").mkdir(exist_ok=True)
            run_dir = directory / "stages" / f"{stage['name']}-{token}"
            if run_dir.exists():
                raise RuntimeError("New run directory already exists; never overwritten")
            argv = [sys.executable, str(root / "scripts/activegs/run_campaign.py"),
                    "--root", str(root), "--worker", token]
            attempt = {"token": token, "stage": stage["name"], "stage_index": state["stage_index"],
                       "gpu": available[0]["index"], "run_dir": str(run_dir),
                       "log": str(directory / "logs" / f"{stage['name']}-{token}.log"),
                       "worker_argv": argv, "status": "starting", "created_at": timestamp()}
            state["attempts"].append(attempt)
            state["active"] = attempt
            state["status"] = "running"
            state.pop("waiting_reason", None)
            state.pop("blocking_job", None)
            save_state(root, state)
            process = spawn_worker(argv, attempt["log"])
            identity, start, code = observe_started_process(process, argv)
            if code is not None:
                raise RuntimeError(f"Worker exited during startup with exit {code}; inspect the stage log")
            # A live child whose argv is not yet observable stays in starting.
            # Its own self-registration or the next tick will recover identity.
            attempt.update(worker=identity, worker_start=start,
                           status="running" if identity else "starting")
            save_state(root, state)
            return state
        except Exception as error:
            return failed(root, state, f"{type(error).__name__}: {error}")


def run_child(root, token, argv, env):
    """Track every live subprocess, including CPU audit/export, before waiting."""
    with campaign_lock(root):
        state = read_state(root)
        active = state["active"]
        if active["token"] != token:
            raise RuntimeError("Campaign attempt changed unexpectedly")
        if source_identity(root) != state["source_identity"]:
            raise RuntimeError("Frozen source changed before subprocess execution")
        # Persist intent before Popen, so a worker crash cannot hide a launched
        # child from another entry point before its PID becomes observable.
        active.update(child_argv=argv, child_launching=True, child_exit_code=None)
        save_state(root, state)
        try:
            process = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL)
        except Exception:
            active["child_launching"] = False  # Popen explicitly did not launch.
            save_state(root, state)
            raise
        identity, start, observed_code = observe_started_process(process, argv)
        active.update(child=identity, child_start=start,
                      child_launching=identity is None and start is None and observed_code is None,
                      child_exit_code=observed_code)
        save_state(root, state)
    code = process.wait()
    with campaign_lock(root):
        state = read_state(root)
        if state["active"]["token"] != token:
            raise RuntimeError("Campaign attempt changed while child was executing")
        state["active"].update(child=None, child_start=None, child_launching=False, child_exit_code=code)
        save_state(root, state)
    if code:
        raise RuntimeError(f"Child failed with exit {code}: {argv[1]}")


def validate_stage(run_dir, stage):
    # Both modules use stdlib only; check() reads real output headers/files again.
    from check_artifacts import check
    from aggregate_benchmark import aggregate
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "benchmark-summary.json").read_text(encoding="utf-8"))
    expected = {(method, seed) for method in stage["methods"] for seed in stage["seeds"]}
    entries = summary.get("experiments", [])
    actual = [(entry.get("method"), entry.get("seed")) for entry in entries]
    if (summary.get("status") != "completed" or summary.get("scene") != "replica/office0"
            or summary.get("methods") != stage["methods"] or summary.get("seeds") != stage["seeds"]
            or len(actual) != len(expected) or set(actual) != expected
            or any(entry.get("status") != "completed" for entry in entries)):
        raise ValueError("Benchmark summary is incomplete or differs from the frozen stage")
    recipe = summary.get("protocol", {})
    for key, value in (("mode", stage["mode"]), ("observations", stage["frames"]),
                       ("prefix", stage["prefix"]), ("seconds", stage["seconds"]),
                       ("candidate_count", 100), ("roi_count", 30), ("sample_points", 500000),
                       ("future_candidate_depth_mask", False)):
        if recipe.get(key) != value:
            raise ValueError(f"Stage protocol differs: {key}")
    experiments = []
    for method, seed in sorted(expected):
        experiment = run_dir / "experiments/benchmark/replica/office0" / method / str(seed)
        check(experiment)
        experiments.append(experiment)
    aggregate(experiments, stage["methods"], stage["seeds"])
    return experiments


def worker(root, token):
    root = Path(root).resolve()
    try:
        with campaign_lock(root):
            state = read_state(root)
            active = state["active"]
            if active is None or active["token"] != token or active["status"] not in ("starting", "running"):
                raise RuntimeError("Worker token is not the active managed attempt")
            identity = proc_identity(os.getpid())
            if identity is None or identity["argv"] != active["worker_argv"]:
                raise RuntimeError("Worker process identity does not match the registered command")
            active["worker"] = identity
            active["worker_start"] = None
            if source_identity(root) != state["source_identity"]:
                raise RuntimeError("Frozen source changed before worker startup")
            stage = STAGES[active["stage_index"]]
            rows = read_gpus()
            if active["gpu"] not in [row["index"] for row in idle_gpus(rows)]:
                # Explicit admission deferral: no benchmark child was launched.
                active.update(status="deferred", finished_at=timestamp(), admission="GPU became occupied before launch")
                state["status"] = "waiting"
                state["waiting_reason"] = "gpu_busy"
                state.pop("blocking_job", None)
                save_state(root, state)
                return 0
            active.update(status="running", started_at=timestamp())
            state.pop("waiting_reason", None)
            state.pop("blocking_job", None)
            save_state(root, state)
        run_dir = Path(active["run_dir"])
        if run_dir.exists():
            raise ValueError("Run directory exists before execution; outputs are never overwritten")
        python = root / ".envs/activegs/bin/python"
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError("Configured Linux ActiveGS Python environment is unavailable")
        run_child(root, token, benchmark_argv(root, stage, active), experiment_environment(root, active["gpu"]))
        experiments = validate_stage(run_dir, stage)
        cpu_env = experiment_environment(root, "")
        if stage["name"] != "smoke":
            run_child(root, token, [sys.executable, str(root / "scripts/activegs/aggregate_benchmark.py"),
                      str(run_dir), "--methods", *stage["methods"], "--expected-seeds", *map(str, stage["seeds"]),
                      "--output", str(run_dir / "paired-results.json")], cpu_env)
            for experiment in experiments:
                method, seed = experiment.parent.name, experiment.name
                output = root / "runs/web-assets" / f"campaign-{stage['name']}-{token}-{method}-s{seed}"
                if output.exists():
                    raise ValueError("Export directory already exists; never replaced")
                run_child(root, token, [str(python), str(root / "scripts/web/export_run.py"), str(run_dir),
                          str(output), "--experiment", str(experiment.relative_to(run_dir))], cpu_env)
                if json.loads((output / "manifest.json").read_text())["status"] != "completed":
                    raise ValueError("Exported manifest is not completed")
        with campaign_lock(root):
            state = read_state(root)
            if source_identity(root) != state["source_identity"]:
                raise RuntimeError("Frozen source changed during execution; results require manual review")
            active = state["active"]
            active.update(status="completed", finished_at=timestamp(), validated_experiments=len(experiments))
            state["stage_index"] += 1
            state["status"] = "completed" if state["stage_index"] == len(STAGES) else "waiting"
            state.pop("waiting_reason", None)
            state.pop("blocking_job", None)
            save_state(root, state)
        return 0
    except Exception as error:
        with campaign_lock(root):
            state = read_state(root)
            if (state.get("active") or {}).get("token") == token:
                failed(root, state, f"{type(error).__name__}: {error}")
        print(f"Campaign failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--tick", action="store_true")
    mode.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.plan:
        result = {"version": VERSION, "status": "plan_only_no_gpu_allocation", "stages": STAGES,
                  "execution_platform": "Linux server", "idle_limits": {"memory_mb_below": 1024, "utilization_max": 5},
                  "state_file": str(location(args.root.resolve()) / "state.json"),
                  "failure_policy": "stop; preserve logs and outputs; no automatic retries"}
    elif args.status:
        result = read_state(args.root.resolve())
    elif args.worker:
        return worker(args.root, args.worker)
    else:
        result = tick(args.root)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
