"""Launcher boundaries and lifecycle tests; never allocate a real GPU."""
import http.client
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import server

IDLE = {"index": 2, "available": True}
BUSY = {"index": 2, "available": False}


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ("runs/web-jobs", "runs/web-assets", "web", ".envs/activegs/bin"):
            (self.root / folder).mkdir(parents=True)
        (self.root / ".envs/activegs/bin/python").touch()

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_occupied_gpu_and_untrusted_parameters(self):
        with self.assertRaises(RuntimeError):
            server.validate_job({"gpu": 2, "budget": 60}, [BUSY])
        for payload in ({"gpu": True, "budget": 60}, {"gpu": "2;echo x", "budget": 60},
                        {"gpu": 2, "budget": True}, {"gpu": 2, "budget": 61},
                        {"gpu": 2, "budget": 60, "command": "arbitrary"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                server.validate_job(payload, [IDLE])

    def test_benchmark_protocol_cannot_change_budget_or_execute_arbitrary_methods(self):
        valid = {"gpu": 2, "protocol": "observations", "method": "defect", "seed": 0, "frames": 60}
        self.assertEqual(server.parse_job(valid, [IDLE])["mode"], "benchmark")
        with self.assertRaises(RuntimeError):
            server.parse_job(valid, [BUSY])
        for key, value in (("method", "main.py; shell"), ("protocol", "time"), ("seed", True),
                           ("seed", 3), ("frames", 20), ("gpu", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                server.parse_job(dict(valid, **{key: value}), [IDLE])
        with self.assertRaises(ValueError):
            server.parse_job(dict(valid, command="arbitrary"), [IDLE])

    def test_benchmark_suite_exports_every_completed_method(self):
        path = self.job_file()
        job = json.loads(path.read_text(encoding="utf-8"))
        job.update(mode="benchmark", protocol="observations", method="suite", seed=1, frames=60, budget=180)
        server.atomic_json(path, job)
        def start(command, **kwargs):
            self.assertEqual(Path(command[1]).name, "run_benchmark.sh")
            self.assertEqual(kwargs["env"]["BENCHMARK_METHODS"], "confidence_nooracle,random_matched,defect")
            self.assertEqual(kwargs["env"]["BENCHMARK_SEEDS"], "1")
            run = Path(kwargs["env"]["RUN_DIR"])
            (run / "logs").mkdir(parents=True)
            for method in server.BENCHMARK_METHODS[:3]:
                result = run / f"experiments/benchmark/replica/office0/{method}/1/final_result.json"
                result.parent.mkdir(parents=True)
                result.write_text("{}")
            return SimpleNamespace(pid=os.getpid(), wait=lambda: 0)
        with patch.object(server, "gpu_status", return_value=([IDLE], None)), \
                patch.object(server.subprocess, "Popen", side_effect=start), \
                patch.object(server.subprocess, "run") as exporter:
            server.worker(self.root, path)
        self.assertEqual(exporter.call_count, 3)
        self.assertTrue(all("--experiment" in call.args[0] for call in exporter.call_args_list))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "completed")

    def job_file(self):
        path = self.root / "runs/web-jobs/test.json"
        server.atomic_json(path, {"id": "test-run", "gpu": 2, "budget": 60,
                                 "status": "starting", "started_unix": 0})
        return path

    def test_worker_rechecks_gpu_before_start(self):
        path = self.job_file()
        with patch.object(server, "gpu_status", return_value=([BUSY], None)), patch.object(server.subprocess, "Popen") as popen:
            server.worker(self.root, path)
        popen.assert_not_called()
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "failed")

    def fake_pipeline(self, code):
        def start(*args, **kwargs):
            run = Path(kwargs["env"]["RUN_DIR"])
            (run / "logs").mkdir(parents=True)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(kwargs["env"]["RECORD_INTERVAL"], "12")
            self.assertEqual(args[0], ["bash", str(self.root / "scripts/activegs/run_original.sh")])
            return SimpleNamespace(pid=os.getpid(), wait=lambda: code)
        return start

    def test_pipeline_failure_is_visible_and_skips_export(self):
        path = self.job_file()
        with patch.object(server, "gpu_status", return_value=([IDLE], None)), \
                patch.object(server.subprocess, "Popen", side_effect=self.fake_pipeline(7)), \
                patch.object(server.subprocess, "run") as export:
            server.worker(self.root, path)
        export.assert_not_called()
        job = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(job["status"], "failed")
        self.assertIn("7", job["message"])

    def test_success_exports_new_run_and_marks_completed(self):
        path = self.job_file()
        with patch.object(server, "gpu_status", return_value=([IDLE], None)), \
                patch.object(server.subprocess, "Popen", side_effect=self.fake_pipeline(0)), \
                patch.object(server.subprocess, "run") as export:
            server.worker(self.root, path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "completed")
        self.assertEqual(export.call_args.args[0][-2:],
                         [str(self.root / "runs/test-run"), str(self.root / "runs/web-assets/test-run")])

    def test_export_failure_does_not_claim_completed(self):
        path = self.job_file()
        with patch.object(server, "gpu_status", return_value=([IDLE], None)), \
                patch.object(server.subprocess, "Popen", side_effect=self.fake_pipeline(0)), \
                patch.object(server.subprocess, "run", side_effect=RuntimeError("export failed")):
            server.worker(self.root, path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "failed")

    def test_http_origin_traversal_and_duplicate_launch(self):
        (self.root / "web/index.html").write_text("test page")
        (self.root / "private.txt").write_text("must not serve")
        with patch.object(server, "ROOT", self.root), \
                patch.object(server, "gpu_status", return_value=([IDLE], None)), \
                patch.object(server.os, "kill", return_value=None), \
                patch.object(server.subprocess, "Popen", return_value=SimpleNamespace(pid=os.getpid())) as popen:
            httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            port = httpd.server_address[1]
            def request(method, path, body=None, headers=None):
                conn = http.client.HTTPConnection("127.0.0.1", port)
                conn.request(method, path, body, headers or {})
                response = conn.getresponse()
                status, content = response.status, response.read()
                conn.close()
                return status, content
            try:
                self.assertEqual(request("GET", "/%2e%2e/private.txt")[0], 403)
                payload = json.dumps({"gpu": 2, "budget": 60})
                self.assertEqual(request("POST", "/api/jobs", payload, {"Content-Type": "application/json"})[0], 403)
                headers = {"Content-Type": "application/json", "X-ViewMend3D": "1", "Origin": "http://untrusted.example"}
                self.assertEqual(request("POST", "/api/jobs", payload, headers)[0], 403)
                headers["Origin"] = f"http://127.0.0.1:{port}"
                self.assertEqual(request("POST", "/api/jobs", payload, headers)[0], 202)
                self.assertEqual(request("POST", "/api/jobs", payload, headers)[0], 409)
                self.assertEqual(popen.call_count, 1)
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
