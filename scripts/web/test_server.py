"""Launcher boundaries and lifecycle tests; never allocate a real GPU."""
import http.client
import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import server

IDLE = {"index": 2, "available": True}
BUSY = {"index": 2, "available": False}


class LauncherTests(unittest.TestCase):
    def test_read_only_http_never_queries_gpu_or_launches_jobs(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'ROOT', Path(folder)), \
                patch.object(server, 'READ_ONLY', True), \
                patch.object(server, 'gpu_status', side_effect=AssertionError('GPU queried')), \
                patch.object(server, 'launch_lock', side_effect=AssertionError('launch intent acquired')), \
                patch.object(server.subprocess, 'Popen', side_effect=AssertionError('worker launched')):
            httpd = server.ThreadingHTTPServer(('127.0.0.1',0), server.Handler)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
            port = httpd.server_address[1]
            try:
                connection = http.client.HTTPConnection('127.0.0.1', port)
                connection.request('GET', '/api/status')
                response = connection.getresponse(); value = json.loads(response.read())
                self.assertEqual(response.status, 200); self.assertTrue(value['read_only'])
                self.assertEqual(value['gpus'], []); self.assertEqual(value['jobs'], [])
                self.assertEqual(value['campaigns'], {}); connection.close()
                connection = http.client.HTTPConnection('127.0.0.1', port)
                connection.request('POST', '/api/jobs', json.dumps({'gpu':2,'budget':60}),
                    {'Content-Type':'application/json','X-ViewMend3D':'1','Origin':f'http://127.0.0.1:{port}'})
                response = connection.getresponse(); response.read()
                self.assertEqual(response.status,403); connection.close()
                self.assertFalse((Path(folder) / 'runs/web-jobs').exists())
            finally:
                httpd.shutdown(); httpd.server_close(); thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ("runs/web-jobs", "runs/web-assets", "web", ".envs/activegs/bin"):
            (self.root / folder).mkdir(parents=True)
        (self.root / ".envs/activegs/bin/python").touch()
        # Admission executes only on Linux. These portable unit tests use
        # a local context; separate Linux race checks exercise real flock.
        self.lock_patch = patch.object(server, "launch_lock", side_effect=lambda root: nullcontext())
        self.lock_patch.start()
        self.addCleanup(self.lock_patch.stop)
        self.observe_patch = patch.object(server, "observe_started_process",
            side_effect=lambda process, argv: ({"pid": process.pid, "starttime": "1", "argv": argv}, None, None))
        self.observe_patch.start()
        self.addCleanup(self.observe_patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    @contextmanager
    def http_service(self):
        with patch.object(server, "ROOT", self.root):
            httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            port = httpd.server_address[1]
            def request(payload):
                conn = http.client.HTTPConnection("127.0.0.1", port)
                conn.request("POST", "/api/jobs", json.dumps(payload),
                             {"Content-Type": "application/json", "X-ViewMend3D": "1",
                              "Origin": f"http://127.0.0.1:{port}"})
                response = conn.getresponse()
                status, content = response.status, json.loads(response.read())
                conn.close()
                return status, content
            try:
                yield request
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join()

    def campaign_file(self, state):
        path = self.root / "runs/campaigns/optimization-v1/state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
        return path

    def campaign_state(self, **changes):
        return {"version": "viewmend-campaign-v1", "status": "waiting", "stage_index": 0,
                "attempts": [], "active": None, "updated_at": "2026-10-06T02:00:00+00:00",
                "gpu_check": {"checked_at": "2026-10-06T01:59:59+00:00"}, **changes}

    def test_campaign_missing_state_never_claims_started(self):
        state = server.campaign_status(self.root)
        self.assertFalse(state["configured"])
        self.assertEqual(state["status"], "not_started")
        self.assertEqual(state["completed_experiments"], 0)
        self.assertFalse((self.root / "runs/campaigns").exists())

    def test_campaign_projection_has_real_counts_and_no_internal_paths(self):
        completed = {"token": "completed-token", "stage": "smoke", "stage_index": 0,
                     "status": "completed", "gpu": 2, "validated_experiments": 3,
                     "run_dir": "/private/run", "worker_argv": ["private", "command"]}
        self.campaign_file(self.campaign_state(stage_index=1, attempts=[completed], active=completed,
                                              source_identity={"private": "source"}))
        state = server.campaign_status(self.root)
        self.assertEqual(state["status"], "waiting")
        self.assertEqual(state["stage"], "observations")
        self.assertEqual(state["completed_experiments"], 3)
        self.assertEqual(state["completed_stages"], 1)
        self.assertNotIn("private", json.dumps(state))
        self.assertIsNone(state["gpu"])
        all_completed = [{"stage": name, "stage_index": index, "status": "completed",
                          "validated_experiments": count, "gpu": 2}
                         for index, (name, count) in enumerate(zip(server.CAMPAIGN_STAGES, server.CAMPAIGN_COUNTS))]
        self.campaign_file(self.campaign_state(status="completed", stage_index=3,
                                              attempts=all_completed, active=all_completed[-1]))
        state = server.campaign_status(self.root)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["completed_experiments"], 27)
        self.assertIsNone(state["stage"])

    def test_campaign_bad_schema_does_not_break_status_or_guess_completion(self):
        for changes in ({"version": "unknown"}, {"stage_index": True},
                        {"status": "completed", "stage_index": 0},
                        {"stage_index": 1}, {"updated_at": "yesterday"},
                        {"status": "running", "active": None}, {"attempts": "invalid"},
                        {"waiting_reason": "unknown"}):
            with self.subTest(changes=changes):
                path = self.campaign_file(self.campaign_state(**changes))
                before = path.read_bytes()
                state = server.campaign_status(self.root)
                self.assertEqual(state["status"], "unavailable")
                self.assertIsNone(state["completed_experiments"])
                self.assertEqual(path.read_bytes(), before)
        path.write_text('{"status":"waiting","status":"completed"}', encoding="utf-8")
        self.assertEqual(server.campaign_status(self.root)["status"], "unavailable")

    def test_campaign_waiting_reason_preserves_other_job_and_gpu_distinction(self):
        for reason in ("managed_web_job", "gpu_busy", "managed_campaign"):
            with self.subTest(reason=reason):
                self.campaign_file(self.campaign_state(waiting_reason=reason))
                self.assertEqual(server.campaign_status(self.root)["waiting_reason"], reason)

    def v2_campaign_file(self, **changes):
        name = "optimization-v2"
        profile = server.get_profile(name)
        state = {"version": profile["version"], "campaign": name,
                 "campaign_spec": profile, "campaign_spec_sha256": server.campaign_spec_sha256(name),
                 "status": "waiting", "stage_index": 0, "attempts": [], "active": None,
                 "updated_at": "2026-10-06T02:00:00+00:00", **changes}
        path = self.root / "runs/campaigns" / name / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
        return path

    def test_v2_projection_uses_five_registered_stage_counts_and_preserves_v1(self):
        profile = server.get_profile("optimization-v2")
        complete = [{"token": f"fixture-v2-{index}", "stage": stage["name"], "stage_index": index, "status": "completed", "gpu": 2,
                     "validated_experiments": len(stage["methods"]) * len(stage["seeds"])}
                    for index, stage in enumerate(profile["stages"])]
        self.v2_campaign_file(stage_index=3, attempts=complete[:3], active=complete[2])
        state = server.campaign_status(self.root, "optimization-v2")
        self.assertEqual(state["completed_experiments"], 15)
        self.assertEqual(state["total_stages"], 5)
        self.assertEqual(state["stage"], "heldout_observations")
        self.assertFalse(server.campaign_status(self.root)["configured"])
        self.v2_campaign_file(status="completed", stage_index=5, attempts=complete, active=complete[-1])
        self.assertEqual(server.campaign_status(self.root, "optimization-v2")["completed_experiments"], 27)

    def test_v2_projection_missing_or_altered_recipe_never_claims_completion(self):
        path = self.v2_campaign_file(campaign_spec_sha256="a" * 64)
        before = path.read_bytes()
        state = server.campaign_status(self.root, "optimization-v2")
        self.assertEqual(state["status"], "unavailable")
        self.assertIsNone(state["completed_experiments"])
        self.assertEqual(path.read_bytes(), before)
        self.v2_campaign_file(status="completed", stage_index=3)
        self.assertEqual(server.campaign_status(self.root, "optimization-v2")["status"], "unavailable")

    def test_http_unknown_v2_launch_intent_blocks_before_gpu_query_and_worker(self):
        self.v2_campaign_file(status="failed", active={"status": "failed", "child_launching": True})
        with patch.object(server, "gpu_status", side_effect=AssertionError("GPU queried")), \
             patch.object(server.subprocess, "Popen", side_effect=AssertionError("worker started")), \
             self.http_service() as request:
            status, _ = request({"gpu": 2, "budget": 60})
        self.assertEqual(status, 409)

    def test_http_live_campaign_blocks_even_when_gpu_query_would_report_idle(self):
        identity = {"pid": 1234, "starttime": "123", "argv": ["python", "campaign_worker.py"]}
        active = {"status": "running", "stage": "smoke", "stage_index": 0, "worker": identity, "gpu": 2}
        self.campaign_file(self.campaign_state(status="running", active=active, attempts=[active]))
        with patch("viewmend3d.launch_coordination.proc_identity", return_value=identity), \
                patch.object(server, "gpu_status", return_value=([IDLE], None)) as query, \
                patch.object(server.subprocess, "Popen") as launch, self.http_service() as request:
            status, body = request({"gpu": 2, "budget": 60})
        self.assertEqual(status, 409)
        self.assertIn("计划", body["error"])
        launch.assert_not_called()
        query.assert_not_called()
        self.assertEqual(list((self.root / "runs/web-jobs").glob("*.json")), [])

    def test_http_shared_lock_covers_intent_spawn_and_identity_sidecar(self):
        held, registered = {"value": False}, {"value": False}
        @contextmanager
        def admission(root):
            self.assertEqual(root, self.root)
            held["value"] = True
            try:
                yield
            finally:
                self.assertTrue(registered["value"])
                held["value"] = False
        def query():
            self.assertTrue(held["value"])
            return [IDLE], None
        def spawn(command, **kwargs):
            self.assertTrue(held["value"])
            self.assertEqual(json.loads(Path(command[-1]).read_text(encoding="utf-8"))["status"], "starting")
            return SimpleNamespace(pid=1234)
        def observe(process, command):
            self.assertTrue(held["value"])
            return {"pid": process.pid, "starttime": "123", "argv": command}, None, None
        original_atomic = server.atomic_json
        def persist(path, value):
            self.assertTrue(held["value"])
            original_atomic(path, value)
            if path.name.endswith(".process.json"):
                registered["value"] = True
        with patch.object(server, "launch_lock", side_effect=admission), \
                patch.object(server, "gpu_status", side_effect=query), \
                patch.object(server.subprocess, "Popen", side_effect=spawn), \
                patch.object(server, "observe_started_process", side_effect=observe), \
                patch.object(server, "atomic_json", side_effect=persist), self.http_service() as request:
            status, body = request({"gpu": 2, "budget": 60})
        self.assertEqual(status, 202)
        self.assertTrue(registered["value"])
        self.assertFalse(held["value"])
        self.assertNotIn("worker_argv", body)
        self.assertEqual(len(list((self.root / "runs/web-jobs").glob("*.process.json"))), 1)

    def test_web_child_releases_launch_lock_before_wait_and_persists_handle(self):
        held = {"value": False}
        path = self.job_file()
        job = json.loads(path.read_text())
        @contextmanager
        def admission(root):
            held["value"] = True
            try:
                yield
            finally:
                held["value"] = False
        def wait():
            self.assertFalse(held["value"])
            saved = json.loads(path.read_text())
            self.assertEqual(saved["child"]["pid"], 1234)
            self.assertEqual(saved["child_argv"], ["bash", "runner.sh"])
            return 7
        with patch.object(server, "launch_lock", side_effect=admission), \
                patch.object(server.subprocess, "Popen", return_value=SimpleNamespace(pid=1234, wait=wait)):
            code = server.run_web_child(self.root, path, job, ["bash", "runner.sh"])
        self.assertEqual(code, 7)
        self.assertIsNone(job["child"])
        self.assertEqual(job["child_exit_code"], 7)
        self.assertFalse(job["child_launching"])

    def test_dead_worker_live_child_blocks_new_web_job_and_status_is_read_only(self):
        path = self.job_file()
        job = json.loads(path.read_text())
        child = {"pid": 4321, "starttime": "123", "argv": ["bash", "runner.sh"]}
        job.update(status="failed", worker=None, child=child, child_launching=False)
        server.atomic_json(path, job)
        server.atomic_json(path.with_suffix(".process.json"),
                           {"identity": None, "start": None, "expected_argv": ["python", "server.py"]})
        before = path.read_bytes()
        with patch("viewmend3d.launch_coordination.proc_identity", return_value=child), \
                patch.object(server, "gpu_status", return_value=([IDLE], None)), \
                patch.object(server.subprocess, "Popen") as launch, self.http_service() as request:
            status, _ = request({"gpu": 2, "budget": 60})
        self.assertEqual(status, 409)
        launch.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        listed = server.jobs(self.root)
        self.assertEqual(len(listed), 1)
        self.assertNotIn("child", listed[0])
        self.assertNotIn("worker_argv", listed[0])
        self.assertEqual(path.read_bytes(), before)

    def test_missing_launch_handle_preserves_active_intent_in_read_only_status(self):
        path = self.job_file()
        before = path.read_bytes()
        listed = server.jobs(self.root)
        self.assertEqual(listed[0]["status"], "starting")
        self.assertEqual(listed[0]["process_state"], "unverified")
        self.assertEqual(path.read_bytes(), before)

    def test_campaign_running_requires_observed_handle_and_keeps_error_plain_text(self):
        active = {"token": "active-token", "stage": "smoke", "stage_index": 0,
                  "status": "running", "gpu": 2,
                  "worker": {"pid": 1234, "starttime": "123", "argv": ["python", "worker.py"]}}
        path = self.campaign_file(self.campaign_state(status="running", active=active, attempts=[active]))
        for observed in ("live", "starting", "unverified"):
            with self.subTest(observed=observed), patch.object(server, "_campaign_process_state", return_value=observed):
                state = server.campaign_status(self.root)
                self.assertEqual(state["status"], "running")
                self.assertEqual(state["process_state"], observed)
        failure = "<img src=x onerror=alert(1)>"
        path.write_text(json.dumps(self.campaign_state(status="failed", error=failure)), encoding="utf-8")
        state = server.campaign_status(self.root)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error"], failure)

    def test_campaign_process_probe_rejects_recycled_pid_and_accepts_start_window(self):
        active = {"worker": {"pid": 1234, "starttime": "123", "argv": ["python", "worker.py"]}}
        def stat(start="123", process_state="S"):
            fields = [process_state] + ["0"] * 18 + [start]
            return "1234 (process name) " + " ".join(fields)
        with patch.object(Path, "read_text", return_value=stat()), \
                patch.object(Path, "read_bytes", return_value=b"python\0worker.py\0"):
            self.assertEqual(server._campaign_process_state(active), "live")
        with patch.object(Path, "read_text", side_effect=[stat(), stat("456")]), \
                patch.object(Path, "read_bytes", return_value=b"python\0worker.py\0"):
            self.assertEqual(server._campaign_process_state(active), "unverified")
        with patch.object(Path, "read_text", return_value=stat(process_state="Z")):
            self.assertEqual(server._campaign_process_state(active), "unverified")
        with patch.object(Path, "read_text", return_value=stat()):
            self.assertEqual(server._campaign_process_state({"worker_start": {"pid": 1234, "starttime": "123"}}), "starting")

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

    def test_guarded_web_method_selects_fixed_recipe_and_refuses_parameter_overrides(self):
        payload = {"gpu": 2, "protocol": "observations", "method": "defect_guarded", "seed": 0, "frames": 60}
        spec = server.parse_job(payload, [IDLE])
        self.assertEqual(spec["recipe"], "optimization-v2")
        self.assertEqual(spec["campaign_spec_sha256"], server.campaign_spec_sha256("optimization-v2"))
        self.assertNotIn("recipe", server.parse_job(dict(payload, method="defect"), [IDLE]))
        for field, value in (("beta", 0.2), ("recipe", "optimization-v1"), ("campaign_spec_sha256", "a" * 64)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                server.parse_job(dict(payload, **{field: value}), [IDLE])

    def test_guarded_web_worker_passes_recipe_and_spec_to_fixed_python_pipeline(self):
        path = self.job_file()
        job = json.loads(path.read_text(encoding="utf-8"))
        job.update(server.parse_job({"gpu": 2, "protocol": "observations", "method": "defect_guarded",
                                    "seed": 1, "frames": 60}, [IDLE]))
        server.atomic_json(path, job)
        def start(command, **kwargs):
            self.assertEqual(command[0], str(self.root / ".envs/activegs/bin/python"))
            if Path(command[1]).name == "run_benchmark.py":
                self.assertEqual(kwargs["env"]["BENCHMARK_METHODS"], "defect_guarded")
                self.assertEqual(kwargs["env"]["BENCHMARK_RECIPE"], "optimization-v2")
                self.assertEqual(kwargs["env"]["BENCHMARK_CAMPAIGN_SPEC_SHA256"], server.campaign_spec_sha256("optimization-v2"))
                run = Path(kwargs["env"]["RUN_DIR"])
                self.assertEqual(command, [str(self.root / ".envs/activegs/bin/python"),
                    str(self.root / "scripts/activegs/run_benchmark.py"),
                    "--upstream", str(self.root / "external/active-gs"), "--run-dir", str(run),
                    "--gpu", "2", "--methods", "defect_guarded", "--seeds", "1",
                    "--frames", "60", "--prefix-frames", "20", "--protocol", "observations",
                    "--budget", "180", "--recipe", "optimization-v2", "--campaign-spec-sha256",
                    server.campaign_spec_sha256("optimization-v2")])
                (run / "logs").mkdir(parents=True)
                result = run / "experiments/benchmark/replica/office0/defect_guarded/1/final_result.json"
                result.parent.mkdir(parents=True)
                result.write_text("{}")
            else:
                self.assertEqual(Path(command[1]), self.root / "scripts/web/export_run.py")
                self.assertIn("--experiment", command)
            return SimpleNamespace(pid=os.getpid(), wait=lambda: 0)
        with patch.object(server, "gpu_status", return_value=([IDLE], None)), \
             patch.object(server.subprocess, "Popen", side_effect=start):
            server.worker(self.root, path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "completed")

    def test_guarded_web_worker_rejects_tampered_recipe_before_pipeline(self):
        path = self.job_file()
        job = json.loads(path.read_text(encoding="utf-8"))
        job.update(server.parse_job({"gpu": 2, "protocol": "observations", "method": "defect_guarded",
                                    "seed": 0, "frames": 60}, [IDLE]))
        job["campaign_spec_sha256"] = "a" * 64
        server.atomic_json(path, job)
        with patch.object(server, "gpu_status", return_value=([IDLE], None)), \
             patch.object(server.subprocess, "Popen", side_effect=AssertionError("pipeline started")):
            server.worker(self.root, path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "failed")

    def test_old_web_job_cannot_inherit_v2_recipe_from_server_environment(self):
        with patch.dict(os.environ, {"BENCHMARK_RECIPE": "optimization-v2", "BENCHMARK_CAMPAIGN_SPEC_SHA256": "a" * 64}):
            env = server.environment(self.root, 2, 180, self.root / "fixture-run")
        self.assertNotIn("BENCHMARK_RECIPE", env)
        self.assertNotIn("BENCHMARK_CAMPAIGN_SPEC_SHA256", env)

    def test_benchmark_suite_exports_every_completed_method(self):
        path = self.job_file()
        job = json.loads(path.read_text(encoding="utf-8"))
        job.update(mode="benchmark", protocol="observations", method="suite", seed=1, frames=60, budget=180)
        server.atomic_json(path, job)
        def start(command, **kwargs):
            self.assertEqual(command[0], str(self.root / ".envs/activegs/bin/python"))
            if Path(command[1]).name == "export_run.py":
                self.assertEqual(Path(command[1]), self.root / "scripts/web/export_run.py")
                return SimpleNamespace(pid=os.getpid(), wait=lambda: 0)
            self.assertEqual(Path(command[1]), self.root / "scripts/activegs/run_benchmark.py")
            self.assertEqual(kwargs["env"]["BENCHMARK_METHODS"], "confidence_nooracle,random_matched,defect")
            self.assertEqual(kwargs["env"]["BENCHMARK_SEEDS"], "1")
            run = Path(kwargs["env"]["RUN_DIR"])
            self.assertEqual(command, [str(self.root / ".envs/activegs/bin/python"),
                str(self.root / "scripts/activegs/run_benchmark.py"),
                "--upstream", str(self.root / "external/active-gs"), "--run-dir", str(run),
                "--gpu", "2", "--methods", "confidence_nooracle", "random_matched", "defect",
                "--seeds", "1", "--frames", "60", "--prefix-frames", "20",
                "--protocol", "observations", "--budget", "180", "--recipe", "optimization-v1"])
            (run / "logs").mkdir(parents=True)
            for method in server.BENCHMARK_METHODS[:3]:
                result = run / f"experiments/benchmark/replica/office0/{method}/1/final_result.json"
                result.parent.mkdir(parents=True)
                result.write_text("{}")
            return SimpleNamespace(pid=os.getpid(), wait=lambda: 0)
        with patch.object(server, "gpu_status", return_value=([IDLE], None)), \
                patch.object(server.subprocess, "Popen", side_effect=start) as launches:
            server.worker(self.root, path)
        exporters = [call.args[0] for call in launches.call_args_list
                     if Path(call.args[0][1]).name == "export_run.py"]
        self.assertEqual(len(exporters), 3)
        self.assertTrue(all("--experiment" in command for command in exporters))
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

    def fake_pipeline(self, code, export_code=0):
        def start(*args, **kwargs):
            if args[0][0] != "bash":
                self.assertEqual(Path(args[0][1]).name, "export_run.py")
                return SimpleNamespace(pid=os.getpid(), wait=lambda: export_code)
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
                patch.object(server.subprocess, "Popen", side_effect=self.fake_pipeline(7)) as launches:
            server.worker(self.root, path)
        self.assertEqual(launches.call_count, 1)
        job = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(job["status"], "failed")
        self.assertIn("7", job["message"])

    def test_success_exports_new_run_and_marks_completed(self):
        path = self.job_file()
        with patch.object(server, "gpu_status", return_value=([IDLE], None)), \
                patch.object(server.subprocess, "Popen", side_effect=self.fake_pipeline(0)) as launches:
            server.worker(self.root, path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "completed")
        self.assertEqual(launches.call_args.args[0][-2:],
                         [str(self.root / "runs/test-run"), str(self.root / "runs/web-assets/test-run")])

    def test_export_failure_does_not_claim_completed(self):
        path = self.job_file()
        with patch.object(server, "gpu_status", return_value=([IDLE], None)), \
                patch.object(server.subprocess, "Popen", side_effect=self.fake_pipeline(0, export_code=7)):
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
                status, content = request("GET", "/api/status")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(content)["campaign"]["status"], "not_started")
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
