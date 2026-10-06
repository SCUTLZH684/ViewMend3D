"""CPU-only checks of v2 registration, frozen plans and cross-campaign admission."""
from contextlib import nullcontext
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/activegs"))
import run_campaign as campaign
from viewmend3d.campaign_profiles import get_profile, campaign_spec_sha256
from viewmend3d.protocol import Protocol

V2 = "optimization-v2"
IDLE = [{"index": 2, "memory_mb": 16, "utilization": 0}]
SOURCE = {"project_commit": "cpu-fixture", "project_source_sha256": "a" * 64}
PROCESS = {"pid": 321, "starttime": "543", "argv": ["fixture-worker"]}


class V2CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.locks = patch.object(campaign, "campaign_lock", lambda root, name=campaign.NAME: nullcontext())
        self.shared = patch.object(campaign, "launch_lock", lambda root: nullcontext())
        self.locks.start()
        self.shared.start()
        self.addCleanup(self.locks.stop)
        self.addCleanup(self.shared.stop)
        self.addCleanup(self.temp.cleanup)

    def save(self, name=V2, **fields):
        state = campaign.read_state(self.root, name)
        state.update(fields)
        campaign.location(self.root, name).mkdir(parents=True, exist_ok=True)
        campaign.save_state(self.root, state, name)
        return state

    def running(self, index=0, **fields):
        stage = get_profile(V2)["stages"][index]
        attempt = {"token": "fixture-v2", "stage": stage["name"], "stage_index": index,
                   "gpu": 2, "run_dir": str(self.root / "fixture-output"),
                   "worker": PROCESS, "worker_argv": PROCESS["argv"], "status": "running", **fields}
        state_status = "failed" if attempt["status"] == "failed" else "running"
        return self.save(status=state_status, stage_index=index, active=attempt,
                         attempts=[attempt], source_identity=SOURCE)

    def prepare_python(self):
        python = self.root / ".envs/activegs/bin/python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("cpu-fixture-not-executed")
        return patch("os.access", return_value=True)

    def test_read_only_plan_has_registered_27_branch_split_and_no_imports(self):
        fresh = self.root / "absent"
        before = set(sys.modules)
        with patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")), \
             patch.object(campaign, "source_identity", side_effect=AssertionError("source queried")), \
             patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(campaign.main(["--root", str(fresh), "--campaign", V2, "--plan"]), 0)
        plan = json.loads(out.getvalue())
        self.assertEqual([len(stage["methods"]) * len(stage["seeds"]) for stage in plan["stages"]], [3, 8, 4, 6, 6])
        self.assertEqual(plan["stages"][3]["seeds"], [3, 4, 5])
        self.assertEqual(plan["stages"][3]["name"], "heldout_observations")
        self.assertEqual(plan["campaign_spec_sha256"], campaign_spec_sha256(V2))
        self.assertFalse(fresh.exists())
        self.assertFalse(any(name == "torch" or name.startswith("habitat") for name in set(sys.modules) - before))

    def test_profile_results_are_independent_and_v1_plan_remains_unchanged(self):
        profile = get_profile(V2)
        original = campaign_spec_sha256(V2)
        profile["stages"][0]["methods"].append("not_registered")
        profile["recipe_spec"]["methods"]["defect_guarded"]["geometry_beta"] = 100
        self.assertEqual(campaign_spec_sha256(V2), original)
        self.assertEqual([stage["name"] for stage in campaign.STAGES], ["smoke", "observations", "time"])
        self.assertNotIn("--recipe", campaign.benchmark_argv(self.root, campaign.STAGES[0], {"run_dir": "fixture", "gpu": 2}))

    def test_v2_launch_preserves_v1_state_and_passes_exact_spec_to_worker_and_benchmark(self):
        self.save(campaign.NAME, status="completed", stage_index=3)
        self.save()
        v1 = (campaign.location(self.root) / "state.json").read_bytes()
        def inspect(pid):
            return {**PROCESS, "argv": spawn.call_args.args[0]}
        with patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign, "spawn_worker", return_value=Mock(pid=321, poll=Mock(return_value=None))) as spawn, \
             patch.object(campaign, "proc_identity", side_effect=inspect):
            state = campaign.tick(self.root, V2)
        self.assertEqual((campaign.location(self.root) / "state.json").read_bytes(), v1)
        self.assertIn(V2, spawn.call_args.args[0])
        self.assertEqual(state["campaign_spec"], get_profile(V2))
        command = campaign.benchmark_argv(self.root, get_profile(V2)["stages"][0], state["active"], V2)
        self.assertEqual(command[command.index("--recipe") + 1], V2)
        self.assertEqual(command[command.index("--campaign-spec-sha256") + 1], campaign_spec_sha256(V2))
        self.assertIn("optimization-v2", state["active"]["run_dir"])

    def test_both_directions_block_foreign_launch_intent_before_gpu_query(self):
        for owner, contender in ((campaign.NAME, V2), (V2, campaign.NAME)):
            with self.subTest(owner=owner):
                if owner == V2:
                    self.running(status="starting", worker=None)
                else:
                    self.save(owner, status="running", active={"status": "starting"})
                self.save(contender, status="not_started", stage_index=0, active=None, attempts=[])
                with patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")), \
                     patch.object(campaign, "spawn_worker", side_effect=AssertionError("spawned")):
                    result = campaign.tick(self.root, contender)
                self.assertEqual(result["waiting_reason"], "managed_campaign")
                self.assertEqual(result["blocking_job"]["campaign"], owner)
                self.assertEqual(result["attempts"], [])
                self.save(owner, status="not_started", stage_index=0, active=None, attempts=[])

    def test_terminal_v2_with_live_child_or_unknown_launch_still_blocks_all(self):
        state = self.running(status="failed", worker=None, child=PROCESS)
        state["status"] = "failed"
        campaign.save_state(self.root, state, V2)
        with patch.object(campaign.coordination, "proc_identity", return_value=PROCESS):
            blocker = campaign.coordination.blocking_campaign(self.root)
        self.assertEqual(blocker["reason"], "live_process")
        state["active"].update(child=None, child_launching=True)
        campaign.save_state(self.root, state, V2)
        self.assertEqual(campaign.coordination.blocking_campaign(self.root)["reason"], "unverified_child_launch")
        with patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")):
            self.assertEqual(campaign.tick(self.root, V2)["live_process"], "unverified_child_launch")

    def test_changed_or_missing_spec_blocks_without_overwriting_metadata(self):
        state = self.save(status="waiting")
        path = campaign.location(self.root, V2) / "state.json"
        for mutate in (lambda s: s.pop("campaign_spec_sha256"),
                       lambda s: s["campaign_spec"]["stages"][3]["seeds"].reverse(),
                       lambda s: s["campaign_spec"]["recipe_spec"]["methods"]["defect_guarded"].update(geometry_beta=0.2)):
            broken = copy.deepcopy(state)
            mutate(broken)
            path.write_text(json.dumps(broken))
            old = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "specification"):
                campaign.read_state(self.root, V2)
            self.assertEqual(path.read_bytes(), old)
            self.assertEqual(campaign.coordination.blocking_campaign(self.root)["reason"], "unverified_metadata")

    def test_forged_final_index_without_complete_stage_history_cannot_advance(self):
        state = self.save(status="waiting")
        state.update(status="completed", stage_index=5)
        path = campaign.location(self.root, V2) / "state.json"
        path.write_text(json.dumps(state))
        with patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")), \
             self.assertRaisesRegex(ValueError, "validation history"):
            campaign.tick(self.root, V2)
        self.assertEqual(json.loads(path.read_text())["attempts"], [])

    def test_v2_smoke_exports_all_new_methods_before_advance(self):
        state = self.running()
        stage = get_profile(V2)["stages"][0]
        experiments = [Path(state["active"]["run_dir"]) / "experiments/benchmark/replica/office0" / method / "0"
                       for method in stage["methods"]]
        commands = []
        def child(root, token, argv, env, name):
            self.assertEqual(name, V2)
            commands.append((argv, env))
            if argv[1].endswith("export_run.py"):
                self.assertEqual(campaign.read_state(root, V2)["stage_index"], 0)
                output = Path(argv[3])
                output.mkdir(parents=True)
                (output / "manifest.json").write_text(json.dumps({"status": "completed"}))
        with self.prepare_python(), patch.object(campaign, "proc_identity", return_value=PROCESS), \
             patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "run_child", side_effect=child), \
             patch.object(campaign, "validate_stage", return_value=experiments):
            self.assertEqual(campaign.worker(self.root, "fixture-v2", V2), 0)
        self.assertEqual(len(commands), 5)
        self.assertEqual(commands[0][1]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertTrue(all(env["CUDA_VISIBLE_DEVICES"] == "" for _, env in commands[1:]))
        self.assertEqual(campaign.read_state(self.root, V2)["stage_index"], 1)

    def test_v2_smoke_export_failure_preserves_stage_and_never_retries(self):
        state = self.running()
        experiment = Path(state["active"]["run_dir"]) / "experiments/benchmark/replica/office0/defect_guarded/0"
        def child(root, token, argv, env, name):
            if argv[1].endswith("export_run.py"):
                raise RuntimeError("fixture export failed")
        with self.prepare_python(), patch.object(campaign, "proc_identity", return_value=PROCESS), \
             patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "run_child", side_effect=child), \
             patch.object(campaign, "validate_stage", return_value=[experiment]), \
             patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(campaign.worker(self.root, "fixture-v2", V2), 1)
        self.assertEqual(campaign.read_state(self.root, V2)["stage_index"], 0)
        with patch.object(campaign, "proc_identity", return_value=None), \
             patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")):
            self.assertEqual(campaign.tick(self.root, V2)["status"], "failed")

    def test_real_json_roundtrip_tracks_worker_child_completion_and_later_failure(self):
        # Exercise the real worker/save_state/run_child functions against real
        # JSON files. Only child execution and reconstruction validation are
        # substituted; no fake GPU process or ML library is launched.
        state = self.running(status="starting", worker=None)
        experiments = [Path(state["active"]["run_dir"]) / "experiments/benchmark/replica/office0" / method / "0"
                       for method in get_profile(V2)["stages"][0]["methods"]]
        snapshots, commands, observed_exits = [], [], []

        def roundtrip():
            stored = campaign.read_state(self.root, V2)
            active = stored["active"]
            historical = next(attempt for attempt in stored["attempts"] if attempt["token"] == active["token"])
            self.assertIsNot(active, historical)
            self.assertEqual(active, historical)
            return stored

        roundtrip()  # The worker will read two independent dictionaries.

        def start(argv, **kwargs):
            intent = roundtrip()
            self.assertTrue(intent["active"]["child_launching"])
            self.assertEqual(intent["active"]["worker"], PROCESS)
            self.assertEqual(intent["active"]["status"], "running")
            self.assertEqual(intent["active"]["child_argv"], argv)
            self.assertIsNone(intent["active"]["child_exit_code"])
            snapshots.append("intent")
            commands.append(argv)
            process = Mock(pid=1000 + len(commands), poll=Mock(return_value=None))

            def wait():
                registered = roundtrip()
                self.assertFalse(registered["active"]["child_launching"])
                self.assertEqual(registered["active"]["child"]["pid"], process.pid)
                self.assertEqual(registered["active"]["child"]["argv"], argv)
                self.assertIsNone(registered["active"]["child_exit_code"])
                snapshots.append("registered")
                if argv[1].endswith("export_run.py"):
                    output = Path(argv[3])
                    output.mkdir(parents=True)
                    (output / "manifest.json").write_text(json.dumps({"status": "completed"}))
                return 0

            process.wait.side_effect = wait
            return process

        def observe(process, argv):
            return {"pid": process.pid, "starttime": str(process.pid), "argv": list(argv)}, None, None

        def frozen_source(root):
            stored = roundtrip()
            if stored["active"].get("child_exit_code") == 0:
                self.assertIsNone(stored["active"]["child"])
                self.assertIsNone(stored["active"]["child_start"])
                self.assertFalse(stored["active"]["child_launching"])
                observed_exits.append(stored["active"]["child_argv"])
            return SOURCE

        with self.prepare_python(), patch.object(campaign, "proc_identity", return_value=PROCESS), \
             patch.object(campaign, "source_identity", side_effect=frozen_source), \
             patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign.subprocess, "Popen", side_effect=start), \
             patch.object(campaign, "observe_started_process", side_effect=observe), \
             patch.object(campaign, "validate_stage", return_value=experiments):
            self.assertEqual(campaign.worker(self.root, "fixture-v2", V2), 0)

        complete = roundtrip()
        self.assertEqual(snapshots, ["intent", "registered"] * 5)
        self.assertEqual(observed_exits, commands)
        self.assertEqual(complete["stage_index"], 1)
        self.assertEqual(complete["status"], "waiting")
        self.assertEqual(complete["active"]["status"], "completed")
        self.assertEqual(complete["active"]["validated_experiments"], 3)
        self.assertIsNone(complete["active"]["child"])
        self.assertFalse(complete["active"]["child_launching"])
        self.assertEqual(complete["active"]["child_exit_code"], 0)
        previous = copy.deepcopy(complete["active"])
        campaign.failed(self.root, complete, "next stage CPU fixture source failure", V2)
        failed = roundtrip()
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["stage_index"], 1)
        self.assertEqual(failed["active"], previous)
        self.assertEqual(failed["attempts"], [previous])

    def test_unknown_active_token_never_writes_state(self):
        self.running()
        path = campaign.location(self.root, V2) / "state.json"
        before = path.read_bytes()
        state = campaign.read_state(self.root, V2)
        self.assertIsNot(state["active"], state["attempts"][0])
        state["active"]["token"] = "unregistered-token"
        with self.assertRaisesRegex(ValueError, "durable history"):
            campaign.save_state(self.root, state, V2)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(campaign.read_state(self.root, V2)["active"]["token"], "fixture-v2")

    def complete_stage(self, index=0):
        stage = get_profile(V2)["stages"][index]
        recipe = Protocol(observations=stage["frames"], prefix=stage["prefix"],
                          mode=stage["mode"], seconds=stage["seconds"], recipe=V2,
                          campaign_spec_sha256=campaign_spec_sha256(V2)).validate().as_dict()
        summary = {"status": "completed", "scene": "replica/office0", "methods": stage["methods"],
                   "seeds": stage["seeds"], "protocol": recipe, "recipe": V2,
                   "recipe_sha256": recipe["recipe_sha256"], "campaign_spec_sha256": campaign_spec_sha256(V2),
                   "experiments": [{"method": method, "seed": seed, "status": "completed"}
                                   for method in stage["methods"] for seed in stage["seeds"]]}
        output = self.root / "fixture-stage"
        output.mkdir()
        (output / "benchmark-summary.json").write_text(json.dumps(summary))
        for method in stage["methods"]:
            for seed in stage["seeds"]:
                path = output / "experiments/benchmark/replica/office0" / method / str(seed)
                path.mkdir(parents=True)
                (path / "protocol.json").write_text(json.dumps({key: summary[key] for key in
                    ("recipe", "recipe_sha256", "campaign_spec_sha256", "protocol")}))
        return stage, output, summary

    def test_exact_recipe_and_heldout_pairs_checked_before_artifact_or_statistics(self):
        stage, output, summary = self.complete_stage(3)
        with patch("check_artifacts.check") as files, patch("aggregate_benchmark.aggregate"):
            self.assertEqual(len(campaign.validate_stage(output, stage, V2)), 6)
        self.assertEqual(files.call_count, 6)
        summary["experiments"].pop()
        (output / "benchmark-summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            campaign.validate_stage(output, stage, V2)

    def test_old_or_mutated_recipe_report_is_rejected_before_stats(self):
        stage, output, summary = self.complete_stage()
        for value in ("optimization-v1", None):
            summary["recipe"] = value
            (output / "benchmark-summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "recipe"):
                campaign.validate_stage(output, stage, V2)
        summary["recipe"] = V2
        summary["protocol"]["recipe_spec"]["methods"]["defect_guarded"]["geometry_beta"] = 0.2
        (output / "benchmark-summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "recipe"):
            campaign.validate_stage(output, stage, V2)

    @unittest.skipUnless(sys.platform == "linux", "Real cross-campaign flock/process race requires Linux")
    def test_real_v1_v2_and_http_admission_register_only_one_owned_cpu_worker(self):
        # The only launched children run these CPU sleep fixtures. GPU queries,
        # reconstruction and the actual server campaign directories are unused.
        scripts = self.root / "scripts/activegs"
        scripts.mkdir(parents=True)
        dummy = scripts / "run_campaign.py"
        dummy.write_text("import time\ntime.sleep(30)\n")
        web_dummy = self.root / "scripts/web/fixture_worker.py"
        web_dummy.parent.mkdir(parents=True)
        web_dummy.write_text("import time\ntime.sleep(30)\n")
        driver = self.root / "fixture_driver.py"
        driver.write_text('''import http.client,json,sys,time,threading
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]); kind=sys.argv[2]; repo=Path(sys.argv[3])
sys.path.insert(0,str(repo/'scripts/activegs')); sys.path.insert(0,str(repo/'scripts/web'))
import run_campaign as campaign
campaign.read_gpus=lambda:[{"index":2,"memory_mb":16,"utilization":0}]
campaign.source_identity=lambda root:{"project_commit":"cpu-fixture"}
(root/('ready-'+kind)).touch()
while not (root/'begin').exists(): time.sleep(.01)
if kind!='http':
 print(json.dumps(campaign.tick(root,kind)))
else:
 import server
 server.ROOT=root; server.__file__=str(root/'scripts/web/fixture_worker.py')
 server.gpu_status=lambda:([{"index":2,"available":True}],None)
 httpd=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
 thread=threading.Thread(target=httpd.serve_forever,daemon=True); thread.start()
 conn=http.client.HTTPConnection('127.0.0.1',httpd.server_address[1])
 conn.request('POST','/api/jobs',json.dumps({"gpu":2,"budget":60}),{"Content-Type":"application/json","X-ViewMend3D":"1"})
 response=conn.getresponse(); print(response.status,response.read().decode()); conn.close()
 httpd.shutdown(); httpd.server_close(); thread.join()
''')
        for folder in ("runs/web-jobs", "runs/web-assets", ".envs/activegs/bin"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        (self.root / ".envs/activegs/bin/python").touch()
        processes = [subprocess.Popen([sys.executable, str(driver), str(self.root), name, str(ROOT)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for name in (campaign.NAME, V2, "http")]
        owned = []
        try:
            deadline = time.monotonic() + 15
            while len(list(self.root.glob("ready-*"))) != 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(list(self.root.glob("ready-*"))), 3)
            (self.root / "begin").touch()
            for process in processes:
                output, error = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, error + output)
            records = []
            for name in (campaign.NAME, V2):
                state = campaign.read_state(self.root, name)
                if state.get("active"):
                    records.append(state["active"])
            records.extend(json.loads(path.read_text()) for path in (self.root / "runs/web-jobs").glob("*.json")
                           if not path.name.endswith(".process.json"))
            for record in records:
                identity = record.get("worker")
                if identity is None and record.get("id"):
                    saved = json.loads((self.root / "runs/web-jobs" / (record["id"] + ".process.json")).read_text())
                    identity = saved.get("identity")
                if identity and campaign.coordination.process_alive(identity):
                    owned.append(identity)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(owned), 1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
            # Only terminate the exact CPU child whose argv names a fixture in
            # this test's private temp root. Never touch server/other-user jobs.
            for identity in owned:
                if (campaign.coordination.process_alive(identity)
                        and identity["argv"][1] in (str(dummy), str(web_dummy))):
                    os.kill(identity["pid"], signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
