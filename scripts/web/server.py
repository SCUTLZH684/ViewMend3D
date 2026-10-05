"""Loopback-only viewer and constrained ActiveGS launcher (Python stdlib).

Serve through SSH forwarding. This is a single-user research tool, not a public
multi-user service. GPU occupancy is checked again immediately before launch.
"""
import argparse
import gzip
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from viewmend3d.launch_coordination import (blocking_campaign, blocking_web_job,
    launch_lock, observe_started_process, proc_identity, proc_start_identity, process_alive)

STAGES = [("preflight.log", "环境与初始观测"), ("data_generation.log", "生成评估视角"),
          ("main.log", "主动采集与重建"), ("mesh_generation.log", "生成三维网格"),
          ("eval.log", "几何评估")]
ACTIVE = {"starting", "running", "exporting"}
BENCHMARK_METHODS = ("confidence_nooracle", "random_matched", "defect", "defect_no_gate", "refine_only")
CAMPAIGN_VERSION = "viewmend-campaign-v1"
CAMPAIGN_STAGES = ("smoke", "observations", "time")
CAMPAIGN_COUNTS = (3, 15, 9)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
START_LOCK = threading.Lock()
STATIC_CACHE = {}
JOB_PUBLIC_FIELDS = {"id", "gpu", "budget", "mode", "protocol", "method", "seed", "frames",
    "record_interval", "status", "started_unix", "message", "wall_seconds", "stage", "log",
    "result_id", "result_ids", "process_state"}


def public_job(job):
    return {key: value for key, value in job.items() if key in JOB_PUBLIC_FIELDS}


def _campaign_timestamp(value):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("invalid campaign timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("campaign timestamp needs timezone")
    return value


def _campaign_identity(value, with_argv):
    if value is None:
        return None
    if not isinstance(value, dict) or type(value.get("pid")) is not int or value["pid"] <= 0:
        raise ValueError("invalid campaign process identity")
    starttime = value.get("starttime")
    if not isinstance(starttime, str) or not starttime.isdigit() or len(starttime) > 32:
        raise ValueError("invalid campaign process starttime")
    result = {"pid": value["pid"], "starttime": starttime}
    if with_argv:
        argv = value.get("argv")
        if (not isinstance(argv, list) or not 1 <= len(argv) <= 100
                or any(not isinstance(item, str) or not item or len(item) > 8192 for item in argv)):
            raise ValueError("invalid campaign process argv")
        result["argv"] = argv
    return result


def _campaign_process_state(active):
    """Observe registered Linux handles without mutating or importing the runner."""
    starting = False
    for kind in ("worker", "child"):
        exact = _campaign_identity(active.get(kind), True)
        start = _campaign_identity(active.get(f"{kind}_start"), False)
        for expected in (exact, start):
            if expected is None:
                continue
            try:
                process = Path("/proc") / str(expected["pid"])
                stat = (process / "stat").read_text()
                fields = stat[stat.rfind(")") + 2:].split()
                if fields[0] == "Z" or fields[19] != expected["starttime"]:
                    continue
                if expected is start:
                    starting = True
                    continue
                argv = (process / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
                observed = [os.fsdecode(item) for item in argv]
                # Read starttime again to avoid validating a recycled PID.
                verify = (process / "stat").read_text()
                verify_fields = verify[verify.rfind(")") + 2:].split()
                if (verify_fields[0] != "Z" and verify_fields[19] == expected["starttime"]
                        and observed == expected["argv"]):
                    return "live"
            except (OSError, IndexError, ValueError):
                continue
    return "starting" if starting else "unverified"


def campaign_status(root):
    """Return a strict, read-only public projection of optimization-v1 state.

    Stored completion is never inferred from a missing process. A stored
    running state additionally reports whether its registered handle is live.
    Internal paths, command lines and source identities are not returned.
    """
    result = {"schema": "viewmend-campaign-status-v1", "configured": False,
              "status": "not_started", "stage": None, "completed_stages": 0,
              "total_stages": 3, "completed_experiments": 0, "total_experiments": 27,
              "gpu": None, "process_state": None, "checked_at": None,
              "updated_at": None, "waiting_reason": None, "error": None}
    path = root / "runs/campaigns/optimization-v1/state.json"
    if not path.exists():
        return result
    result["configured"] = True
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("campaign state exceeds size limit")
        def reject_constant(value):
            raise ValueError("invalid JSON numeric constant")
        def unique_object(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate JSON object key")
                value[key] = item
            return value
        state = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant,
                           object_pairs_hook=unique_object)
        if not isinstance(state, dict) or state.get("version") != CAMPAIGN_VERSION:
            raise ValueError("unsupported campaign schema")
        status, index = state.get("status"), state.get("stage_index")
        if status not in ("not_started", "waiting", "running", "failed", "completed"):
            raise ValueError("invalid campaign status")
        if type(index) is not int or not 0 <= index <= 3:
            raise ValueError("invalid campaign stage index")
        if (status == "completed") != (index == 3) or (status == "not_started" and index != 0):
            raise ValueError("campaign status and stage disagree")
        attempts = state.get("attempts")
        active = state.get("active")
        if not isinstance(attempts, list) or (active is not None and not isinstance(active, dict)):
            raise ValueError("invalid campaign attempts")
        completed = {}
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise ValueError("invalid campaign attempt")
            step, name = attempt.get("stage_index"), attempt.get("stage")
            if type(step) is not int or not 0 <= step < 3 or name != CAMPAIGN_STAGES[step]:
                raise ValueError("invalid campaign attempt stage")
            if attempt.get("status") not in ("starting", "running", "completed", "deferred", "failed"):
                raise ValueError("invalid campaign attempt status")
            if attempt["status"] == "completed":
                count = attempt.get("validated_experiments")
                if type(count) is not int or count != CAMPAIGN_COUNTS[step] or step in completed:
                    raise ValueError("invalid validated experiment count")
                completed[step] = count
        if set(completed) != set(range(index)):
            raise ValueError("completed campaign stages lack matching validation records")
        if active:
            active_index = active.get("stage_index")
            if (type(active_index) is not int or not 0 <= active_index < 3
                    or active.get("stage") != CAMPAIGN_STAGES[active_index]):
                raise ValueError("invalid active campaign stage")
            gpu = active.get("gpu")
            if type(gpu) is not int or gpu < 0:
                raise ValueError("invalid campaign GPU")
        if status == "running":
            if not active or active.get("stage_index") != index or active.get("status") not in ("starting", "running"):
                raise ValueError("running campaign lacks an active attempt")
            result["gpu"] = active["gpu"]
            result["process_state"] = _campaign_process_state(active)
        check = state.get("gpu_check")
        if check is not None and not isinstance(check, dict):
            raise ValueError("invalid campaign GPU check")
        error = state.get("error")
        if error is not None and (not isinstance(error, str) or len(error) > 8192):
            raise ValueError("invalid campaign error")
        waiting_reason = state.get("waiting_reason")
        if waiting_reason not in (None, "gpu_busy", "managed_web_job"):
            raise ValueError("invalid campaign waiting reason")
        result.update(status=status, stage=CAMPAIGN_STAGES[index] if index < 3 else None,
                      completed_stages=index, completed_experiments=sum(completed.values()),
                      checked_at=_campaign_timestamp((check or {}).get("checked_at")),
                      updated_at=_campaign_timestamp(state.get("updated_at")),
                      waiting_reason=waiting_reason if status == "waiting" else None, error=error)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        result.update(status="unavailable", stage=None, completed_stages=None,
                      completed_experiments=None, gpu=None, process_state=None,
                      error=f"计划状态无法读取：{type(error).__name__}。请检查服务端记录。")
    return result


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def gpu_status():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=8, check=True)
        gpus = []
        for line in result.stdout.splitlines():
            index, name, used, total, util = [part.strip() for part in line.split(",")]
            used, total, util = int(used), int(total), int(util)
            gpus.append({"index": int(index), "name": name, "used_mb": used, "total_mb": total,
                         "utilization": util, "available": used < 1024 and util <= 5})
        return gpus, None
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return [], f"无法读取 GPU 状态：{type(error).__name__}"


def validate_job(payload, gpus):
    if not isinstance(payload, dict) or set(payload) != {"gpu", "budget"}:
        raise ValueError("只接受 gpu 和 budget 两个参数")
    gpu, budget = payload["gpu"], payload["budget"]
    if type(gpu) is not int or type(budget) is not int or budget not in (60, 180, 300):
        raise ValueError("GPU 编号必须为整数，任务预算仅支持 60 / 180 / 300 秒")
    selected = next((item for item in gpus if item["index"] == gpu), None)
    if selected is None:
        raise ValueError("所选 GPU 不存在或 GPU 状态不可用")
    if not selected["available"]:
        raise RuntimeError("所选 GPU 已被占用，请等待空闲卡；没有启动实验")
    return gpu, budget


def parse_job(payload, gpus):
    """Accept only the original demo or the frozen 60-event benchmark protocol."""
    if isinstance(payload, dict) and set(payload) == {"gpu", "budget"}:
        gpu, budget = validate_job(payload, gpus)
        return {"gpu": gpu, "budget": budget, "mode": "original"}
    if not isinstance(payload, dict) or set(payload) != {"gpu", "protocol", "method", "seed", "frames"}:
        raise ValueError("需要原始预算参数或完整的固定观测协议参数")
    if (payload["protocol"] != "observations" or payload["method"] not in BENCHMARK_METHODS + ("suite",)
            or type(payload["seed"]) is not int or payload["seed"] not in (0, 1, 2)
            or type(payload["frames"]) is not int or payload["frames"] != 60):
        raise ValueError("公平协议只支持 60 次更新、种子 0/1/2 和列出的算法")
    gpu, _ = validate_job({"gpu": payload["gpu"], "budget": 180}, gpus)
    return dict(payload, gpu=gpu, mode="benchmark", budget=180)


def tail(path, limit=14000):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        content = stream.read(limit).decode("utf-8", errors="replace")
    # tqdm uses carriage returns; keep the final update on each logical line.
    return ANSI.sub("", "\n".join(line.split("\r")[-1] for line in content.split("\n")))


def jobs(root):
    result = []
    for path in sorted((root / "runs/web-jobs").glob("*.json"), reverse=True):
        if path.name.endswith(".process.json"):
            continue
        job = json.loads(path.read_text(encoding="utf-8"))
        if job["status"] in ACTIVE:
            # Status polling is read-only. A missing PID or observation gap
            # cannot establish that a launch intent or child is terminal.
            handles = [job.get("worker"), job.get("child")]
            starts = [job.get("worker_start"), job.get("child_start")]
            sidecar = path.with_suffix(".process.json")
            if sidecar.exists():
                try:
                    registration = json.loads(sidecar.read_text(encoding="utf-8"))
                    handles.append(registration.get("identity"))
                    starts.append(registration.get("start"))
                except (OSError, ValueError, TypeError):
                    pass
            live = any(handle and process_alive(handle) for handle in handles)
            starting = any(start and proc_start_identity(start["pid"]) == start for start in starts)
            job["process_state"] = "live" if live else "starting" if starting else "unverified"
        stage = "准备启动"
        log_path = root / "runs" / job["id"] / "logs/runner.log"
        for name, label in STAGES:
            candidate = root / "runs" / job["id"] / "logs" / name
            if candidate.exists():
                log_path, stage = candidate, label
        benchmark_log = root / "runs" / job["id"] / "logs/benchmark.log"
        if job.get("mode") == "benchmark":
            worker_log = root / "runs/web-jobs" / f"{job['id']}.log"
            log_path, stage = (benchmark_log if benchmark_log.exists() else worker_log), "公共前缀、算法对照与几何评估"
        if job["status"] == "exporting" or (root / "runs" / job["id"] / "logs/export.log").exists():
            stage = "导出浏览器预览"
            log_path = root / "runs" / job["id"] / "logs/export.log"
        if job.get("process_state") == "unverified":
            stage = "进程状态待核查；未推断完成或重新启动"
        elif job.get("process_state") == "starting":
            stage = "启动进程身份确认中"
        job["stage"] = stage
        job["log"] = tail(log_path)
        job["wall_seconds"] = int(time.time() - job["started_unix"]) if job["status"] in ACTIVE else job.get("wall_seconds", 0)
        result.append(public_job(job))
    return sorted(result, key=lambda job: job["started_unix"], reverse=True)


def run_web_child(root, job_path, job, command, **kwargs):
    """Persist a child launch intent and live handle before waiting for exit."""
    with launch_lock(root):
        job.update(child=None, child_start=None, child_argv=command,
                   child_exit_code=None, child_launching=True)
        atomic_json(job_path, job)
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, **kwargs)
        except OSError:
            # Popen itself failed: this explicit path never created a child.
            job["child_launching"] = False
            atomic_json(job_path, job)
            raise
        job.update(child_pid=process.pid, child_start=proc_start_identity(process.pid))
        atomic_json(job_path, job)
        identity, start, observed_code = observe_started_process(process, command)
        job.update(child=identity, child_start=start,
                   child_launching=False if identity or observed_code is not None else True)
        atomic_json(job_path, job)
    # Never hold the cross-entry-point lock while the experiment/export runs.
    code = process.wait()
    job.update(child=None, child_start=None, child_launching=False, child_exit_code=code)
    atomic_json(job_path, job)
    return code


def environment(root, gpu, budget, run):
    env = os.environ.copy()
    env.update(ACTIVEGS_ROOT=str(root / "external/active-gs"),
               ACTIVEGS_PYTHON=str(root / ".envs/activegs/bin/python"), RUN_DIR=str(run),
               GPU=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu), BUDGET=str(budget),
               RECORD_INTERVAL=str(budget // 5), OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               MPLBACKEND="Agg", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    env["LD_LIBRARY_PATH"] = ":".join([str(root / ".sysroot/usr/lib/x86_64-linux-gnu"),
                                      str(root / ".envs/activegs/lib"), env.get("LD_LIBRARY_PATH", "")])
    return env


def worker(root, job_path):
    job = json.loads(job_path.read_text(encoding="utf-8"))
    run = root / "runs" / job["id"]
    # Check occupancy at execution time as well as at the HTTP request boundary.
    try:
        with launch_lock(root):
            if blocking_campaign(root):
                raise RuntimeError("优化实验计划正在执行或启动记录待核查；没有启动网页实验")
            gpus, error = gpu_status()
            if error:
                raise RuntimeError(error)
            request = ({key: job[key] for key in ("gpu", "protocol", "method", "seed", "frames")}
                       if job.get("mode") == "benchmark" else {"gpu": job["gpu"], "budget": job["budget"]})
            parse_job(request, gpus)
            worker_argv = job.get("worker_argv") or [sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path)]
            job.update(status="running", worker_pid=os.getpid(), worker=proc_identity(os.getpid()),
                       worker_start=proc_start_identity(os.getpid()), worker_argv=worker_argv)
            atomic_json(job_path, job)
        env = environment(root, job["gpu"], job["budget"], run)
        runner = "run_original.sh"
        if job.get("mode") == "benchmark":
            runner = "run_benchmark.sh"
            methods = ",".join(BENCHMARK_METHODS[:3]) if job["method"] == "suite" else job["method"]
            env.update(BENCHMARK_METHODS=methods, BENCHMARK_SEEDS=str(job["seed"]),
                       BENCHMARK_FRAMES="60", BENCHMARK_PREFIX="20", BENCHMARK_PROTOCOL="observations",
                       BENCHMARK_BUDGET="180")
        # The original runner requires a directory that does not yet exist.
        runner_log = root / "runs/web-jobs" / f"{job['id']}.log"
        with runner_log.open("w") as log:
            code = run_web_child(root, job_path, job, ["bash", str(root / "scripts/activegs" / runner)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        if run.exists():
            shutil.copyfile(runner_log, run / "logs/runner.log")
        if code != 0:
            raise RuntimeError(f"实验流水线退出码 {code}，请查看阶段日志")
        job.update(status="exporting")
        atomic_json(job_path, job)
        with (run / "logs/export.log").open("w") as log:
            experiments = sorted((run / "experiments/benchmark/replica/office0").glob("*/*/final_result.json")) if job.get("mode") == "benchmark" else []
            if job.get("mode") == "benchmark" and not experiments:
                raise RuntimeError("对照流水线未生成指标，没有可导出的完成结果")
            result_ids = []
            for result in experiments or [None]:
                identifier = job["id"] if result is None else f"{job['id']}-{result.parent.parent.name}-s{result.parent.name}"
                command = [env["ACTIVEGS_PYTHON"], str(root / "scripts/web/export_run.py"),
                           str(run), str(root / "runs/web-assets" / identifier)]
                if result is not None:
                    command.extend(["--experiment", str(result.parent)])
                code = run_web_child(root, job_path, job, command, env=env, stdout=log, stderr=subprocess.STDOUT)
                if code:
                    raise RuntimeError(f"预览导出退出码 {code}，请查看日志")
                result_ids.append(identifier)
        job["result_ids"] = result_ids
        job["result_id"] = next((identifier for identifier in result_ids if "-defect-s" in identifier), result_ids[0])
        job.update(status="completed", message="重建、几何评估和浏览器预览已完成")
    except Exception as error:
        job.update(status="failed", message=str(error))
    job["wall_seconds"] = int(time.time() - job["started_unix"])
    atomic_json(job_path, job)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"{self.log_date_time_string()} {format % args}", flush=True)

    def valid_host(self):
        host = urlsplit("http://" + self.headers.get("Host", "")).hostname
        return host in ("localhost", "127.0.0.1", "::1")

    def json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.valid_host():
            return self.json({"error": "仅支持本机或 SSH 转发访问"}, 403)
        path = unquote(urlsplit(self.path).path)
        if path == "/api/status":
            gpus, error = gpu_status()
            available_runs = []
            for manifest in sorted((ROOT / "runs/web-assets").glob("*/manifest.json"), reverse=True):
                # Avoid repeatedly reading the full trajectory in the status poll.
                available_runs.append({"id": manifest.parent.name,
                                       "manifest": f"/assets/{manifest.parent.name}/manifest.json",
                                       "summary": f"/assets/{manifest.parent.name}/summary.json" if (manifest.parent / "summary.json").exists() else None})
            return self.json({"gpus": gpus, "gpu_error": error, "jobs": jobs(ROOT),
                              "runs": available_runs, "method": "ActiveGS confidence", "scene": "Replica office0",
                              "benchmark_methods": list(BENCHMARK_METHODS), "benchmark_frames": 60,
                              "campaign": campaign_status(ROOT)})
        if path.startswith("/api/"):
            return self.json({"error": "接口不存在"}, 404)
        base = ROOT / "runs/web-assets" if path.startswith("/assets/") else ROOT / "web"
        relative = path[len("/assets/"):] if path.startswith("/assets/") else path.lstrip("/") or "index.html"
        target = (base / relative).resolve()
        try:
            target.relative_to(base.resolve())
        except ValueError:
            return self.json({"error": "非法文件路径"}, 403)
        if not target.is_file():
            return self.json({"error": "文件不存在"}, 404)
        self.send_response(200)
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
        compressed = None
        if "gzip" in self.headers.get("Accept-Encoding", "") and target.suffix in (".js", ".css", ".html", ".json", ".ply"):
            cache_key = (str(target), target.stat().st_mtime_ns)
            compressed = STATIC_CACHE.get(cache_key)
            if compressed is None:
                compressed = gzip.compress(target.read_bytes(), compresslevel=6)
                STATIC_CACHE[cache_key] = compressed
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(compressed) if compressed is not None else target.stat().st_size))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if compressed is not None:
            self.wfile.write(compressed)
            return
        with target.open("rb") as stream:
            while True:
                chunk = stream.read(128 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if (not self.valid_host() or self.headers.get("X-ViewMend3D") != "1"
                or (origin and origin != "http://" + self.headers.get("Host", ""))):
            return self.json({"error": "请求来源无效"}, 403)
        if urlsplit(self.path).path != "/api/jobs":
            return self.json({"error": "接口不存在"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 or not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ValueError("需要小于 1 KiB 的 JSON 请求")
            payload = json.loads(self.rfile.read(length))
            # Both entry points reserve launch intent under this same Linux
            # flock. Hold it until the worker registration has been persisted.
            with launch_lock(ROOT), START_LOCK:
                if blocking_campaign(ROOT):
                    raise RuntimeError("优化实验计划正在执行或启动记录待核查，请等待完成")
                if blocking_web_job(ROOT):
                    raise RuntimeError("已有网页实验正在执行或启动记录待核查，请等待完成")
                gpus, error = gpu_status()
                if error:
                    raise RuntimeError(error)
                spec = parse_job(payload, gpus)
                gpu, budget = spec["gpu"], spec["budget"]
                if not (ROOT / ".envs/activegs/bin/python").is_file():
                    raise RuntimeError("服务器尚未配置 ActiveGS 环境")
                identifier = datetime.now(timezone.utc).strftime("web-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
                job = {**spec, "id": identifier, "record_interval": budget // 5,
                       "status": "starting", "started_unix": time.time(), "message": "准备执行原始 baseline"}
                if spec["mode"] == "benchmark":
                    job["message"] = "准备执行固定观测协议；前 20 次为共享 confidence 前缀"
                job_path = ROOT / "runs/web-jobs" / f"{identifier}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path)]
                job["worker_argv"] = command
                atomic_json(job_path, job)
                try:
                    with (ROOT / "runs/web-jobs" / f"{identifier}-worker.log").open("w") as log:
                        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as error:
                    job.update(status="failed", message=f"无法启动实验进程：{type(error).__name__}")
                    atomic_json(job_path, job)
                    raise RuntimeError(job["message"]) from error
                identity, start, code = observe_started_process(process, command)
                # The worker owns job.json after spawn. Parent registration
                # lives in a separate atomic sidecar to avoid overwriting it.
                atomic_json(job_path.with_suffix(".process.json"),
                            {"identity": identity, "start": start, "expected_argv": command})
                job["worker_pid"] = process.pid
                if code is not None:
                    observed = json.loads(job_path.read_text(encoding="utf-8"))
                    if observed["status"] in ACTIVE:
                        observed.update(status="failed", message=f"启动进程已退出（{code}），没有验收完成记录")
                        atomic_json(job_path, observed)
                    job = observed
            return self.json(public_job(job), 202)
        except (ValueError, json.JSONDecodeError) as error:
            return self.json({"error": str(error)}, 400)
        except RuntimeError as error:
            return self.json({"error": str(error)}, 409)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    (ROOT / "runs/web-jobs").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs/web-assets").mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(ROOT, args.worker)
    else:
        # Prevent two control servers from launching concurrent jobs into this workspace.
        import fcntl
        lock = (ROOT / "runs/web-jobs/server.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"ViewMend3D: http://127.0.0.1:{args.port}", flush=True)
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
