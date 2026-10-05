"""CPU fixtures for orchestration; no reconstruction/GPU correctness claims."""
from contextlib import contextmanager, nullcontext
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
import subprocess

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/activegs"
sys.path.insert(0, str(SCRIPTS))
import run_campaign as campaign

BUSY = [{"index": 2, "memory_mb": 20000, "utilization": 0}]
IDLE = [{"index": 2, "memory_mb": 10, "utilization": 5}]
SOURCE = {"project_commit": "frozen-fixture", "project_source_sha256": "a" * 64,
          "upstream_commit": "upstream-fixture", "upstream_source_sha256": "b" * 64}
PROCESS = {"pid": 123, "starttime": "9876", "argv": ["fixture-worker"]}


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        campaign.location(self.root).mkdir(parents=True)
        self.lock = patch.object(campaign, "campaign_lock", lambda _: nullcontext())
        self.lock.start()
        self.launch = patch.object(campaign, "launch_lock", lambda _: nullcontext())
        self.launch.start()

    def tearDown(self):
        self.lock.stop()
        self.launch.stop()
        self.temp.cleanup()

    def state(self, **fields):
        state = campaign.read_state(self.root)
        state.update(fields)
        campaign.save_state(self.root, state)
        return state

    def running(self, status="running"):
        attempt = {"token": "fixture-token", "stage": "smoke", "stage_index": 0,
                   "gpu": 2, "run_dir": str(self.root / "fixture-output"),
                   "worker": PROCESS.copy(), "worker_argv": PROCESS["argv"], "status": status}
        return self.state(status="running", active=attempt, attempts=[attempt], source_identity=SOURCE)

    def test_plan_and_status_are_read_only_and_do_not_query_gpu(self):
        fresh = self.root / "fresh"
        with patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")), \
             patch.object(campaign, "source_identity", side_effect=AssertionError("source queried")), \
             patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(campaign.main(["--root", str(fresh), "--plan"]), 0)
            self.assertEqual(campaign.main(["--root", str(fresh), "--status"]), 0)
        self.assertFalse(fresh.exists())

    def test_busy_gpu_tick_is_waiting_without_importing_torch_or_spawning(self):
        before = set(sys.modules)
        with patch.object(campaign, "read_gpus", return_value=BUSY), \
             patch.object(campaign, "spawn_worker", side_effect=AssertionError("spawned")), \
             patch.object(campaign, "source_identity", side_effect=AssertionError("source queried")):
            result = campaign.tick(self.root)
        self.assertEqual(result["status"], "waiting")
        self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in set(sys.modules) - before))
        self.assertEqual(result["attempts"], [])

    def test_alive_worker_is_not_restarted_even_if_status_is_old(self):
        self.running()
        with patch.object(campaign, "proc_identity", return_value=PROCESS), \
             patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")):
            self.assertEqual(campaign.tick(self.root)["live_process"], "worker")

    def test_live_orphan_child_prevents_relaunch(self):
        state = self.running()
        child = {"pid": 456, "starttime": "9", "argv": ["fixture-benchmark"]}
        state["active"]["child"] = child
        campaign.save_state(self.root, state)
        with patch.object(campaign, "proc_identity", side_effect=lambda pid: child if pid == 456 else None), \
             patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")):
            self.assertEqual(campaign.tick(self.root)["live_process"], "child")

    def test_pid_reuse_does_not_count_as_managed_worker(self):
        self.running()
        reused = {**PROCESS, "starttime": "different"}
        with patch.object(campaign, "proc_identity", return_value=reused), \
             patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")):
            result = campaign.tick(self.root)
        self.assertEqual(result["status"], "failed")
        self.assertIn("without verified completion", result["error"])

    def test_failed_campaign_never_retries(self):
        self.state(status="failed", error="fixture child failure")
        with patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")), \
             patch.object(campaign, "spawn_worker", side_effect=AssertionError("spawned")):
            self.assertEqual(campaign.tick(self.root)["status"], "failed")

    def test_launch_records_identity_unique_directory_and_exact_frozen_smoke(self):
        def fingerprint(pid):
            return {**PROCESS, "argv": spawn.call_args.args[0]}
        with patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign, "spawn_worker", return_value=Mock(pid=123, poll=Mock(return_value=None))) as spawn, \
             patch.object(campaign, "proc_identity", side_effect=fingerprint):
            state = campaign.tick(self.root)
        self.assertEqual(state["active"]["gpu"], 2)
        self.assertFalse(Path(state["active"]["run_dir"]).exists())
        self.assertIn("--worker", spawn.call_args.args[0])
        command = campaign.benchmark_argv(self.root, campaign.STAGES[0], state["active"])
        self.assertEqual(command[command.index("--frames") + 1], "6")
        self.assertEqual(command[command.index("--prefix-frames") + 1], "2")
        self.assertEqual(state["source_identity"], SOURCE)
        self.assertEqual(campaign.read_state(self.root)["attempts"][0]["worker"]["argv"], spawn.call_args.args[0])

    def test_changed_source_stops_before_launching_next_stage(self):
        self.state(status="waiting", stage_index=1, source_identity=SOURCE)
        with patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "source_identity", return_value={**SOURCE, "project_commit": "new"}), \
             patch.object(campaign, "spawn_worker", side_effect=AssertionError("spawned")):
            state = campaign.tick(self.root)
        self.assertEqual(state["status"], "failed")
        self.assertIn("source changed", state["error"])

    def test_next_stage_source_change_preserves_completed_attempt_history(self):
        state = self.running("completed")
        state.update(status="waiting", stage_index=1)
        state["active"].update(finished_at="fixture-completed", validated_experiments=3)
        campaign.save_state(self.root, state)
        completed = dict(state["active"])
        with patch.object(campaign, "proc_identity", return_value=None), \
             patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "source_identity", return_value={**SOURCE, "project_commit": "changed"}), \
             patch.object(campaign, "spawn_worker", side_effect=AssertionError("spawned")):
            result = campaign.tick(self.root)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage_index"], 1)
        self.assertEqual(result["active"], completed)
        self.assertEqual(result["attempts"], [completed])
        self.assertIn("source changed", result["error"])

    def test_next_stage_gpu_probe_failure_preserves_completed_attempt_history(self):
        state = self.running("completed")
        state.update(status="waiting", stage_index=1)
        state["active"].update(finished_at="fixture-completed", validated_experiments=3)
        campaign.save_state(self.root, state)
        completed = dict(state["active"])
        with patch.object(campaign, "proc_identity", return_value=None), \
             patch.object(campaign, "read_gpus", side_effect=RuntimeError("fixture GPU probe error")), \
             patch.object(campaign, "spawn_worker", side_effect=AssertionError("spawned")):
            result = campaign.tick(self.root)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage_index"], 1)
        self.assertEqual(result["active"], completed)
        self.assertEqual(campaign.read_state(self.root)["attempts"], [completed])
        self.assertIn("GPU probe error", result["error"])

    def test_campaign_failure_preserves_deferred_attempt_without_started_benchmark(self):
        state = self.running("deferred")
        state.update(status="waiting")
        deferred = dict(state["active"])
        with patch.object(campaign, "proc_identity", return_value=None), \
             patch.object(campaign, "read_gpus", side_effect=RuntimeError("fixture GPU probe error")):
            campaign.save_state(self.root, state)
            result = campaign.tick(self.root)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage_index"], 0)
        self.assertEqual(result["active"], deferred)
        self.assertEqual(result["attempts"], [deferred])

    def test_starting_worker_is_recovered_from_exact_command(self):
        self.running("starting")
        with patch.object(campaign, "proc_identity", return_value=None), \
             patch.object(campaign, "find_worker", return_value=PROCESS), \
             patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")):
            self.assertEqual(campaign.tick(self.root)["active"]["status"], "running")

    def test_worker_rechecks_gpu_and_defers_before_any_child(self):
        self.running()
        with patch.object(campaign, "proc_identity", return_value=PROCESS), \
             patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign, "read_gpus", return_value=BUSY), \
             patch.object(campaign, "run_child", side_effect=AssertionError("child started")):
            code = campaign.worker(self.root, "fixture-token")
        self.assertEqual(code, 0)
        state = campaign.read_state(self.root)
        self.assertEqual(state["status"], "waiting")
        self.assertEqual(state["active"]["status"], "deferred")
        self.assertEqual(state["stage_index"], 0)

    def prepare_worker_env(self):
        python = self.root / ".envs/activegs/bin/python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("fixture-not-executed")
        return patch("os.access", return_value=True)

    def test_worker_nonzero_child_marks_failed_without_advancing(self):
        self.running()
        with self.prepare_worker_env(), patch.object(campaign, "proc_identity", return_value=PROCESS), \
             patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "run_child", side_effect=RuntimeError("fixture exit 7")), \
             patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(campaign.worker(self.root, "fixture-token"), 1)
        state = campaign.read_state(self.root)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stage_index"], 0)
        self.assertIn("exit 7", state["attempts"][0]["error"])

    def test_smoke_advances_only_after_artifact_validation(self):
        self.running()
        with self.prepare_worker_env(), patch.object(campaign, "proc_identity", return_value=PROCESS), \
             patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "run_child") as child, \
             patch.object(campaign, "validate_stage", return_value=[Path("fixture")] * 3) as validate:
            self.assertEqual(campaign.worker(self.root, "fixture-token"), 0)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(validate.call_count, 1)
        state = campaign.read_state(self.root)
        self.assertEqual(state["stage_index"], 1)
        self.assertEqual(state["active"]["status"], "completed")
        self.assertEqual(state["status"], "waiting")

    def test_artifact_validation_failure_stops_later_stages(self):
        self.running()
        with self.prepare_worker_env(), patch.object(campaign, "proc_identity", return_value=PROCESS), \
             patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "run_child"), \
             patch.object(campaign, "validate_stage", side_effect=ValueError("fixture missing mesh")), \
             patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(campaign.worker(self.root, "fixture-token"), 1)
        self.assertEqual(campaign.read_state(self.root)["stage_index"], 0)
        self.assertEqual(campaign.read_state(self.root)["status"], "failed")

    def test_stage_plan_contains_full_methods_seeds_and_time_budget(self):
        stages = campaign.STAGES
        self.assertEqual([stage["name"] for stage in stages], ["smoke", "observations", "time"])
        self.assertEqual(len(stages[1]["methods"]) * len(stages[1]["seeds"]), 15)
        self.assertEqual(len(stages[2]["methods"]) * len(stages[2]["seeds"]), 9)
        self.assertEqual(stages[2]["mode"], "time")
        self.assertEqual(stages[2]["seconds"], 180)

    def test_formal_worker_exports_every_requested_pair_with_cuda_hidden(self):
        state = self.running()
        state["active"].update(stage="observations", stage_index=1)
        state["stage_index"] = 1
        campaign.save_state(self.root, state)
        experiments = [Path(state["active"]["run_dir"]) / "experiments/benchmark/replica/office0" / method / str(seed)
                       for method in campaign.STAGES[1]["methods"] for seed in campaign.STAGES[1]["seeds"]]
        commands = []
        def child(root, token, argv, env):
            commands.append((argv, env))
            if argv[1].endswith("export_run.py"):
                output = Path(argv[3])
                output.mkdir(parents=True)
                (output / "manifest.json").write_text(json.dumps({"status": "completed"}))
        with self.prepare_worker_env(), patch.object(campaign, "proc_identity", return_value=PROCESS), \
             patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "run_child", side_effect=child), \
             patch.object(campaign, "validate_stage", return_value=experiments):
            self.assertEqual(campaign.worker(self.root, "fixture-token"), 0)
        self.assertEqual(len(commands), 17)  # Benchmark, aggregate, all 15 exports.
        self.assertEqual(commands[0][1]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertTrue(all(env["CUDA_VISIBLE_DEVICES"] == "" for _, env in commands[1:]))
        self.assertEqual(campaign.read_state(self.root)["stage_index"], 2)
        self.assertEqual(campaign.read_state(self.root)["status"], "waiting")

    def test_cpu_export_failure_stops_campaign_after_reconstruction(self):
        state = self.running()
        state["active"].update(stage="time", stage_index=2)
        state["stage_index"] = 2
        campaign.save_state(self.root, state)
        experiment = Path(state["active"]["run_dir"]) / "experiments/benchmark/replica/office0/defect/0"
        def child(root, token, argv, env):
            if argv[1].endswith("export_run.py"):
                raise RuntimeError("fixture export failed")
        with self.prepare_worker_env(), patch.object(campaign, "proc_identity", return_value=PROCESS), \
             patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign, "read_gpus", return_value=IDLE), \
             patch.object(campaign, "run_child", side_effect=child), \
             patch.object(campaign, "validate_stage", return_value=[experiment]), \
             patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(campaign.worker(self.root, "fixture-token"), 1)
        self.assertEqual(campaign.read_state(self.root)["stage_index"], 2)
        self.assertEqual(campaign.read_state(self.root)["status"], "failed")

    def test_run_child_persists_live_process_before_waiting(self):
        self.running()
        child_identity = {"pid": 456, "starttime": "1234", "argv": ["fixture-child", "fixture-program"]}
        process = Mock(pid=456)
        def wait():
            self.assertEqual(campaign.read_state(self.root)["active"]["child"], child_identity)
            return 0
        process.wait.side_effect = wait
        with patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign.subprocess, "Popen", return_value=process), \
             patch.object(campaign, "proc_identity", return_value=child_identity):
            campaign.run_child(self.root, "fixture-token", ["fixture-child", "fixture-program"], {})
        self.assertIsNone(campaign.read_state(self.root)["active"]["child"])
        self.assertEqual(campaign.read_state(self.root)["active"]["child_exit_code"], 0)

    def test_child_launch_intent_is_durable_before_popen_and_cleared_on_explicit_failure(self):
        self.running()
        command = ["fixture-child", "fixture-program"]
        def cannot_launch(*args, **kwargs):
            active = campaign.read_state(self.root)["active"]
            self.assertTrue(active["child_launching"])
            self.assertEqual(active["child_argv"], command)
            self.assertIsNone(active["child_exit_code"])
            raise OSError("fixture spawn failure")
        with patch.object(campaign, "source_identity", return_value=SOURCE), \
             patch.object(campaign.subprocess, "Popen", side_effect=cannot_launch):
            with self.assertRaisesRegex(OSError, "spawn failure"):
                campaign.run_child(self.root, "fixture-token", command, {})
        self.assertFalse(campaign.read_state(self.root)["active"]["child_launching"])

    def test_terminal_campaign_unknown_child_launch_window_blocks_web(self):
        state = self.running("failed")
        state["status"] = "failed"
        state["active"].update(worker=None, child=None, child_launching=True,
                               child_argv=["fixture-child"], child_exit_code=None)
        campaign.save_state(self.root, state)
        blocker = campaign.coordination.blocking_campaign(self.root)
        self.assertEqual(blocker["reason"], "unverified_child_launch")

    def test_launch_observation_waits_through_none_and_exec_argv_transition(self):
        process = Mock(pid=123, poll=Mock(return_value=None))
        transition = {**PROCESS, "argv": ["exec-transition"]}
        with patch.object(campaign, "proc_identity", side_effect=[None, transition, PROCESS]), \
             patch.object(campaign.time, "monotonic", return_value=0), \
             patch.object(campaign.time, "sleep"):
            observed, start, code = campaign.observe_started_process(process, PROCESS["argv"])
        self.assertEqual(observed, PROCESS)
        self.assertIsNone(start)
        self.assertIsNone(code)
        self.assertEqual(process.poll.call_count, 2)

    def test_live_unobserved_launch_remains_starting_and_is_not_restarted(self):
        process = Mock(pid=123, poll=Mock(return_value=None))
        start_identity = {"pid": 123, "starttime": "9876"}
        with patch.object(campaign, "proc_identity", return_value=None), \
             patch.object(campaign, "proc_start_identity", return_value=start_identity), \
             patch.object(campaign.time, "monotonic", side_effect=[0, 1]):
            observed, start, code = campaign.observe_started_process(process, PROCESS["argv"])
        self.assertIsNone(observed)
        self.assertIsNone(code)
        self.assertEqual(start, start_identity)
        state = self.running("starting")
        state["active"].update(worker=None, worker_start=start_identity)
        campaign.save_state(self.root, state)
        with patch.object(campaign, "proc_identity", return_value=None), \
             patch.object(campaign, "proc_start_identity", return_value=start_identity), \
             patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")):
            result = campaign.tick(self.root)
        self.assertEqual(result["active"]["status"], "starting")
        self.assertEqual(result["live_process"], "worker_starting")

    def test_launch_termination_uses_exit_code_instead_of_missing_argv(self):
        process = Mock(pid=123, poll=Mock(return_value=7))
        with patch.object(campaign, "proc_identity", return_value=None):
            observed, start, code = campaign.observe_started_process(process, PROCESS["argv"])
        self.assertIsNone(observed)
        self.assertIsNone(start)
        self.assertEqual(code, 7)

    def test_cpu_postprocessing_environment_hides_all_gpus(self):
        env = campaign.experiment_environment(self.root, "")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertIn(".sysroot", env["LD_LIBRARY_PATH"])

    def write_web_job(self, status, **fields):
        folder = self.root / "runs/web-jobs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "fixture-job.json"
        path.write_text(json.dumps({"status": status, **fields}), encoding="utf-8")
        return path

    def test_campaign_rejects_active_web_intent_before_querying_gpu(self):
        self.write_web_job("starting")
        with patch.object(campaign, "read_gpus", side_effect=AssertionError("GPU queried")), \
             patch.object(campaign, "spawn_worker", side_effect=AssertionError("spawned")):
            state = campaign.tick(self.root)
        self.assertEqual(state["status"], "waiting")
        self.assertEqual(state["waiting_reason"], "managed_web_job")
        self.assertEqual(state["blocking_job"]["owner"], "web")
        self.assertEqual(state["attempts"], [])

    def test_campaign_reports_gpu_wait_after_web_job_finishes(self):
        path = self.write_web_job("running")
        state = campaign.tick(self.root)
        self.assertEqual(state["waiting_reason"], "managed_web_job")
        path.write_text(json.dumps({"status": "completed"}))
        with patch.object(campaign, "read_gpus", return_value=BUSY):
            state = campaign.tick(self.root)
        self.assertEqual(state["waiting_reason"], "gpu_busy")
        self.assertNotIn("blocking_job", state)

    def test_shared_launch_lock_precedes_campaign_state_lock(self):
        events = []
        @contextmanager
        def shared(_):
            events.append("shared_enter")
            yield
            events.append("shared_exit")
        @contextmanager
        def inner(_):
            events.append("state_enter")
            yield
            events.append("state_exit")
        with patch.object(campaign, "launch_lock", shared), patch.object(campaign, "campaign_lock", inner), \
             patch.object(campaign, "read_gpus", return_value=BUSY):
            campaign.tick(self.root)
        self.assertEqual(events, ["shared_enter", "state_enter", "state_exit", "shared_exit"])

    def test_terminal_web_job_with_live_child_still_blocks(self):
        self.write_web_job("completed", child=PROCESS)
        with patch.object(campaign.coordination, "proc_identity", return_value=PROCESS):
            blocker = campaign.coordination.blocking_web_job(self.root)
        self.assertEqual(blocker["reason"], "live_process")
        self.assertEqual(blocker["process"], "child")

    def test_terminal_web_job_with_unknown_child_launch_window_blocks(self):
        self.write_web_job("failed", child_launching=True, child_exit_code=None)
        blocker = campaign.coordination.blocking_web_job(self.root)
        self.assertEqual(blocker["reason"], "unverified_child_launch")

    def test_legacy_interrupted_job_ignores_recycled_pid_but_blocks_actual_worker(self):
        path = self.write_web_job("interrupted")
        path.with_suffix(".pid").write_text("123")
        foreign = {**PROCESS, "argv": ["python", "-c", "other task"]}
        with patch.object(campaign.coordination, "proc_identity", return_value=foreign):
            self.assertIsNone(campaign.coordination.blocking_web_job(self.root))
        actual = {**PROCESS, "argv": ["python", str((self.root / "scripts/web/server.py").resolve()),
                                       "--worker", str(path.resolve())]}
        with patch.object(campaign.coordination, "proc_identity", return_value=actual):
            blocker = campaign.coordination.blocking_web_job(self.root)
        self.assertEqual(blocker["reason"], "live_process")
        self.assertEqual(blocker["process"], "legacy_worker")

    def test_orphan_sidecar_blocks_when_original_job_metadata_is_missing(self):
        path = self.write_web_job("completed")
        path.with_suffix(".process.json").write_text(json.dumps({"identity": PROCESS}))
        path.unlink()
        blocker = campaign.coordination.blocking_web_job(self.root)
        self.assertEqual(blocker["reason"], "orphan_process_metadata")

    def test_terminal_campaign_with_live_handle_blocks_web_admission(self):
        state = self.running("failed")
        state["status"] = "failed"
        state["active"].update(worker=None, child=PROCESS)
        campaign.save_state(self.root, state)
        with patch.object(campaign.coordination, "proc_identity", return_value=PROCESS):
            blocker = campaign.coordination.blocking_campaign(self.root)
        self.assertEqual(blocker["reason"], "live_process")
        self.assertEqual(blocker["process"], "child")

    def test_unknown_or_malformed_job_state_conservatively_blocks(self):
        path = self.write_web_job("unknown-fixture")
        self.assertEqual(campaign.coordination.blocking_web_job(self.root)["reason"], "unknown_status")
        path.write_text('{"status":"starting","status":"failed"}')
        self.assertEqual(campaign.coordination.blocking_web_job(self.root)["reason"], "unverified_metadata")
        self.state(status="unexpected-state")
        self.assertEqual(campaign.coordination.blocking_campaign(self.root)["reason"], "unknown_status")

    def stage_files(self):
        from viewmend3d.protocol import Protocol
        from test_artifacts import write_output_chain
        stage = campaign.STAGES[0]
        recipe = Protocol(observations=6, prefix=2).as_dict()
        entries = [{"method": method, "seed": 0, "status": "completed"} for method in stage["methods"]]
        summary = {"status": "completed", "scene": "replica/office0", "methods": stage["methods"],
                   "seeds": [0], "experiments": entries, "protocol": recipe}
        output = self.root / "complete-fixture"
        output.mkdir()
        (output / "benchmark-summary.json").write_text(json.dumps(summary))
        for method in stage["methods"]:
            experiment = output / "experiments/benchmark/replica/office0" / method / "0"
            experiment.mkdir(parents=True)
            results = {"step": [2, 4, 6], "time": [2, 4, 6], "path_length": [0, 0, 0],
                       "mesh_accuracy": [1, 1, 1], "mesh_completion": [2, 2, 2],
                       "mesh_completion_ratio": [80, 80, 80], "mesh_chamfer_distance": [0.015] * 3}
            write_output_chain(experiment, results)
        return output, summary

    def test_fresh_artifact_check_reads_actual_files_despite_completed_summary(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        output, _ = self.stage_files()
        with patch("aggregate_benchmark.aggregate", return_value={}) as audit:
            self.assertEqual(len(campaign.validate_stage(output, campaign.STAGES[0])), 3)
            self.assertEqual(audit.call_count, 1)
        (output / "experiments/benchmark/replica/office0/defect/0/map/mesh_006.ply").unlink()
        with patch("aggregate_benchmark.aggregate", side_effect=AssertionError("Should fail fresh files first")):
            with self.assertRaisesRegex(ValueError, "Missing/empty artifact"):
                campaign.validate_stage(output, campaign.STAGES[0])

    def test_forged_completed_summary_cannot_skip_pairs_or_change_budget(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        output, summary = self.stage_files()
        summary["experiments"].pop()
        (output / "benchmark-summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            campaign.validate_stage(output, campaign.STAGES[0])
        summary["experiments"].append({"method": "refine_only", "seed": 0, "status": "completed"})
        summary["protocol"]["prefix"] = 3
        (output / "benchmark-summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            campaign.validate_stage(output, campaign.STAGES[0])

    @unittest.skipUnless(sys.platform == "linux", "Actual /proc validation runs on Linux server only")
    def test_actual_live_cpu_process_and_exact_argv_identity(self):
        command = [sys.executable, "-c", "import time; time.sleep(10)"]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        try:
            identity, _, code = campaign.observe_started_process(process, command)
            self.assertIsNone(code)
            self.assertEqual(identity["argv"], command)
            self.assertTrue(campaign.process_alive(identity))
            self.assertFalse(campaign.process_alive({**identity, "starttime": "never-the-same"}))
            self.assertEqual(campaign.find_worker(command), identity)
        finally:
            process.terminate()  # Only this CPU test's own subprocess.
            process.wait(timeout=10)
        self.assertIsNone(campaign.proc_identity(process.pid))

    @unittest.skipUnless(sys.platform == "linux", "Actual flock/tick validation runs on Linux server only")
    def test_parallel_real_ticks_with_fake_nvidia_smi_preserve_one_waiting_state(self):
        fake_bin = self.root / "fixture-bin"
        fake_bin.mkdir()
        executable = fake_bin / "nvidia-smi"
        executable.write_text("#!/bin/sh\nprintf '2, 20000, 0\\n'\n")
        executable.chmod(0o755)
        env = {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", "")}
        command = [sys.executable, str(SCRIPTS / "run_campaign.py"), "--root", str(self.root), "--tick"]
        processes = [subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True) for _ in range(3)]
        for process in processes:
            output, errors = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, errors)
            self.assertEqual(json.loads(output)["status"], "waiting")
        state = campaign.read_state(self.root)
        self.assertEqual(state["status"], "waiting")
        self.assertEqual(state["attempts"], [])


if __name__ == "__main__":
    unittest.main()
