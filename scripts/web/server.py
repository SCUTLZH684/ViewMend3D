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
STAGES = [("preflight.log", "环境与初始观测"), ("data_generation.log", "生成评估视角"),
          ("main.log", "主动采集与重建"), ("mesh_generation.log", "生成三维网格"),
          ("eval.log", "几何评估")]
ACTIVE = {"starting", "running", "exporting"}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
START_LOCK = threading.Lock()
STATIC_CACHE = {}


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
        job = json.loads(path.read_text(encoding="utf-8"))
        if job["status"] in ACTIVE:
            pid_file = path.with_suffix(".pid")
            pid = job.get("worker_pid") or (int(pid_file.read_text()) if pid_file.exists() else None)
            try:
                if pid is None and time.time() - job["started_unix"] < 30:
                    continue
                os.kill(pid or -999999, 0)
            except ProcessLookupError:
                job.update(status="interrupted", message="实验进程已退出，请查看日志")
                atomic_json(path, job)
        stage = "准备启动"
        log_path = root / "runs" / job["id"] / "logs/runner.log"
        for name, label in STAGES:
            candidate = root / "runs" / job["id"] / "logs" / name
            if candidate.exists():
                log_path, stage = candidate, label
        if job["status"] == "exporting" or (root / "runs" / job["id"] / "logs/export.log").exists():
            stage = "导出浏览器预览"
            log_path = root / "runs" / job["id"] / "logs/export.log"
        job["stage"] = stage
        job["log"] = tail(log_path)
        job["wall_seconds"] = int(time.time() - job["started_unix"]) if job["status"] in ACTIVE else job.get("wall_seconds", 0)
        result.append(job)
    return sorted(result, key=lambda job: job["started_unix"], reverse=True)


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
        gpus, error = gpu_status()
        if error:
            raise RuntimeError(error)
        validate_job({"gpu": job["gpu"], "budget": job["budget"]}, gpus)
        job.update(status="running", worker_pid=os.getpid())
        atomic_json(job_path, job)
        env = environment(root, job["gpu"], job["budget"], run)
        # The original runner requires a directory that does not yet exist.
        runner_log = root / "runs/web-jobs" / f"{job['id']}.log"
        with runner_log.open("w") as log:
            process = subprocess.Popen(["bash", str(root / "scripts/activegs/run_original.sh")],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            job["pipeline_pid"] = process.pid
            atomic_json(job_path, job)
            code = process.wait()
        if run.exists():
            shutil.copyfile(runner_log, run / "logs/runner.log")
        if code != 0:
            raise RuntimeError(f"原始流水线退出码 {code}，请查看阶段日志")
        job.update(status="exporting")
        atomic_json(job_path, job)
        with (run / "logs/export.log").open("w") as log:
            subprocess.run([env["ACTIVEGS_PYTHON"], str(root / "scripts/web/export_run.py"),
                            str(run), str(root / "runs/web-assets" / job["id"])],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
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
                                       "manifest": f"/assets/{manifest.parent.name}/manifest.json"})
            return self.json({"gpus": gpus, "gpu_error": error, "jobs": jobs(ROOT),
                              "runs": available_runs, "method": "ActiveGS confidence", "scene": "Replica office0"})
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
            with START_LOCK:
                if any(job["status"] in ACTIVE for job in jobs(ROOT)):
                    raise RuntimeError("已有实验正在执行，请等待完成")
                gpus, error = gpu_status()
                if error:
                    raise RuntimeError(error)
                gpu, budget = validate_job(payload, gpus)
                if not (ROOT / ".envs/activegs/bin/python").is_file():
                    raise RuntimeError("服务器尚未配置 ActiveGS 环境")
                identifier = datetime.now(timezone.utc).strftime("web-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
                job = {"id": identifier, "gpu": gpu, "budget": budget, "record_interval": budget // 5,
                       "status": "starting", "started_unix": time.time(), "message": "准备执行原始 baseline"}
                job_path = ROOT / "runs/web-jobs" / f"{identifier}.json"
                atomic_json(job_path, job)
                with (ROOT / "runs/web-jobs" / f"{identifier}-worker.log").open("w") as log:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path)],
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                # The worker owns subsequent updates; keep the PID separate to avoid a write race.
                (job_path.with_suffix(".pid")).write_text(str(process.pid))
                job["worker_pid"] = process.pid
            return self.json(job, 202)
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
