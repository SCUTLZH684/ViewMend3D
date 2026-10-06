"""Standard-library launch coordination shared by campaign and web entry points.

Admission holds ``runs/launch.lock`` until its intent and process handle are
durable. Locks cover registration/admission only; they are released before
waiting for reconstruction/export. Occupancy inspection is read-only; unknown
metadata blocks admission and never changes another job's state.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

from .campaign_profiles import CAMPAIGN_NAMES, get_profile, validate_profile_state

WEB_ACTIVE = {"starting", "running", "exporting"}
WEB_TERMINAL = {"failed", "completed", "interrupted"}
CAMPAIGN_STATES = {"not_started", "waiting", "running", "failed", "completed"}
ATTEMPT_STATES = {"starting", "running", "deferred", "failed", "completed"}


@contextmanager
def launch_lock(root):
    if sys.platform != "linux":
        raise RuntimeError("Shared GPU launch admission requires the Linux server")
    import fcntl
    directory = Path(root) / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "launch.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def proc_start_identity(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError("Process PID must be a positive integer")
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "starttime": fields[19]}
    except FileNotFoundError:
        return None


def proc_identity(pid):
    try:
        start = proc_start_identity(pid)
        if start is None:
            return None
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        if not argv or argv == [b""] or proc_start_identity(pid) != start:
            return None
        return {**start, "argv": [os.fsdecode(part) for part in argv]}
    except FileNotFoundError:
        return None


def process_alive(identity, inspect=None):
    return bool(identity and (inspect or proc_identity)(identity["pid"]) == identity)


def observe_started_process(process, argv, timeout=1.0, inspect=None, inspect_start=None):
    inspect = inspect or proc_identity
    inspect_start = inspect_start or proc_start_identity
    deadline = time.monotonic() + timeout
    while True:
        identity = inspect(process.pid)
        if identity and identity["argv"] == argv:
            return identity, None, None
        code = process.poll()
        if code is not None:
            return None, None, code
        if time.monotonic() >= deadline:
            return None, inspect_start(process.pid), None
        time.sleep(0.02)


def _read_json(path):
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Job metadata exceeds size limit")
    def reject_constant(value):
        raise ValueError("Non-finite JSON constant")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate job metadata key")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant,
                       object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("Job metadata must be an object")
    return value


def _identity(value, argv):
    if value is None:
        return None
    if (not isinstance(value, dict) or type(value.get("pid")) is not int or value["pid"] <= 0
            or not isinstance(value.get("starttime"), str) or not value["starttime"].isdigit()):
        raise ValueError("Invalid stored process identity")
    expected = {"pid": value["pid"], "starttime": value["starttime"]}
    if argv:
        command = value.get("argv")
        if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("Invalid stored process command")
        expected["argv"] = command
    return expected


def _live_handles(record):
    for kind in ("worker", "child"):
        exact = _identity(record.get(kind), True)
        start = _identity(record.get(f"{kind}_start"), False)
        if process_alive(exact):
            return kind
        if start and proc_start_identity(start["pid"]) == start:
            return f"{kind}_starting"
    return None


def _web_command(argv, root, job_path):
    server = str((Path(root) / "scripts/web/server.py").resolve())
    job = str(job_path.resolve())
    # Require an actual Python script argument, not incidental text in -c code.
    return (len(argv) >= 4 and argv[1] == server and any(
        value == "--worker" and index + 1 < len(argv) and argv[index + 1] == job
        for index, value in enumerate(argv)))


def blocking_web_job(root):
    root = Path(root).resolve()
    directory = root / "runs/web-jobs"
    if not directory.exists():
        return None
    paths = sorted(path for path in directory.glob("*.json") if not path.name.endswith(".process.json"))
    for path in paths:
        base = {"owner": "web", "id": path.stem}
        try:
            job = _read_json(path)
            status = job.get("status")
            if status not in WEB_ACTIVE | WEB_TERMINAL:
                return {**base, "reason": "unknown_status"}
            live = _live_handles(job)
            if live:
                return {**base, "reason": "live_process", "process": live, "status": status}
            if job.get("child_launching") is True:
                return {**base, "reason": "unverified_child_launch", "status": status}
            if job.get("child_launching") not in (None, False):
                return {**base, "reason": "unverified_metadata"}
            sidecar = path.with_suffix(".process.json")
            if sidecar.exists():
                saved = _read_json(sidecar)
                exact = _identity(saved.get("identity"), True)
                start = _identity(saved.get("start"), False)
                if exact and _web_command(exact["argv"], root, path) and process_alive(exact):
                    return {**base, "reason": "live_process", "process": "worker", "status": status}
                if start and proc_start_identity(start["pid"]) == start:
                    return {**base, "reason": "live_process", "process": "worker_starting", "status": status}
            pid_file = path.with_suffix(".pid")
            pid = job.get("worker_pid")
            if pid is None and pid_file.exists():
                pid = int(pid_file.read_text().strip())
            if pid is not None:
                observed = proc_identity(pid)
                if observed and _web_command(observed["argv"], root, path):
                    return {**base, "reason": "live_process", "process": "legacy_worker", "status": status}
            if status in WEB_ACTIVE:
                return {**base, "reason": "active_intent", "status": status}
        except (OSError, ValueError, TypeError, IndexError, KeyError):
            return {**base, "reason": "unverified_metadata"}
    for sidecar in sorted(directory.glob("*.process.json")):
        path = sidecar.with_name(sidecar.name.removesuffix(".process.json") + ".json")
        if not path.exists():
            # The original job's terminal state and possible child are unknown.
            return {"owner": "web", "id": path.stem, "reason": "orphan_process_metadata"}
    # Recover a still-live legacy worker whose JSON disappeared, without
    # confusing a recycled PID in an obsolete .pid file with this task.
    for pid_file in sorted(directory.glob("*.pid")):
        path = pid_file.with_suffix(".json")
        if path.exists():
            continue
        try:
            observed = proc_identity(int(pid_file.read_text().strip()))
            if observed and _web_command(observed["argv"], root, path):
                return {"owner": "web", "id": path.stem, "reason": "live_orphan_worker"}
        except (OSError, ValueError, TypeError, IndexError, KeyError):
            return {"owner": "web", "id": path.stem, "reason": "unverified_metadata"}
    return None


def _blocking_campaign(root, name):
    path = Path(root) / "runs/campaigns" / name / "state.json"
    if not path.exists():
        return None
    base = {"owner": "campaign", "campaign": name}
    try:
        state = _read_json(path)
        status = state.get("status")
        if state.get("version") != get_profile(name)["version"] or status not in CAMPAIGN_STATES:
            return {**base, "reason": "unknown_status"}
        validate_profile_state(state, name)
        active = state.get("active")
        if active is not None:
            if not isinstance(active, dict) or active.get("status") not in ATTEMPT_STATES:
                return {**base, "reason": "unverified_metadata"}
            live = _live_handles(active)
            if live:
                return {**base, "reason": "live_process", "process": live, "status": status}
            if active.get("child_launching") is True:
                return {**base, "reason": "unverified_child_launch", "status": status}
            if active.get("child_launching") not in (None, False):
                return {**base, "reason": "unverified_metadata"}
            if active["status"] in ("starting", "running"):
                return {**base, "reason": "active_intent", "status": status}
        if status == "running":
            return {**base, "reason": "active_intent", "status": status}
        return None
    except (OSError, ValueError, TypeError, IndexError, KeyError):
        return {**base, "reason": "unverified_metadata"}


def blocking_campaign(root, exclude=None):
    """Inspect every registered campaign; never infer exit from missing state.

    The caller must hold launch_lock when it uses this for GPU admission. A
    controller excludes itself only after checking its own handles and intent.
    Unknown/unverifiable metadata in another campaign conservatively blocks.
    """
    if exclude is not None and exclude not in CAMPAIGN_NAMES:
        raise ValueError("Only a registered campaign may be excluded")
    for name in CAMPAIGN_NAMES:
        if name != exclude:
            blocker = _blocking_campaign(root, name)
            if blocker:
                return blocker
    return None
