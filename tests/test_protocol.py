"""CPU checks for experimental validity; none launch the reconstruction model."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from viewmend3d.benchmark import ObservationOnlySimulator, gradient_refinement_only, require_idle_gpu
from viewmend3d.protocol import Protocol, domain_seed


class ProtocolChecks(unittest.TestCase):
    def test_formal_and_smoke_checkpoints(self):
        self.assertEqual(Protocol().validate().checkpoints, (20, 40, 60))
        self.assertEqual(Protocol(observations=6, prefix=2).validate().checkpoints, (2, 4, 6))
        with self.assertRaises(ValueError):
            Protocol(prefix=60).validate()

    def test_time_budget_is_independent_of_sixty_event_limit(self):
        protocol = Protocol(mode="time").validate()
        self.assertEqual(protocol.event_cap, 10000)
        self.assertEqual(protocol.as_dict()["time_checkpoints_seconds"], [60, 120, 180])
        self.assertIsNone(protocol.as_dict()["fixed_observation_budget"])
        self.assertEqual(protocol.checkpoints, (20,))

    def test_independent_reproducible_domains(self):
        seeds = {domain_seed(seed, domain, event)
                 for seed in (0, 1, 2) for domain in ("planning", "sensor", "mapping", "evaluation")
                 for event in range(1, 61)}
        self.assertEqual(len(seeds), 3 * 4 * 60)
        self.assertEqual(domain_seed(0, "sensor", 1), domain_seed(0, "sensor", 1))

    def test_gpu_resident_allocation_is_not_free(self):
        output = types.SimpleNamespace(stdout="0, 19361, 0\n1, 100, 0\n")
        with patch("viewmend3d.benchmark.subprocess.run", return_value=output):
            with self.assertRaises(RuntimeError):
                require_idle_gpu(0)
            self.assertEqual(require_idle_gpu(1)["index"], 1)

    def test_candidate_and_gt_simulator_queries_blocked(self):
        calls = []
        backend = types.SimpleNamespace(resolution=[512, 512], intrinsic=None, depth_range=[0, 5],
                                        bbox=[[-1] * 3, [1] * 3], scene_name="test",
                                        simulate=lambda pose: calls.append(pose) or {"depth": 1})
        simulator = ObservationOnlySimulator(backend)
        with self.assertRaises(RuntimeError):
            simulator.simulate("candidate")
        with simulator.acquire_selected_view():
            with self.assertRaises(RuntimeError):
                simulator.simulate("candidate", valid_mask_only=True)
            with self.assertRaises(RuntimeError):
                simulator.simulate("candidate", require_gt=True)
            self.assertEqual(simulator.simulate("selected"), {"depth": 1})
        with self.assertRaises(RuntimeError):
            simulator.simulate("after")
        self.assertEqual(calls, ["selected"])
        self.assertEqual(simulator.observation_calls, 1)

    def test_refinement_does_not_fake_new_view_and_restores_on_failure(self):
        supports = []
        original = lambda: supports.append(1)
        gaussian = types.SimpleNamespace(post_processing=original)
        with self.assertRaises(RuntimeError):
            with gradient_refinement_only(gaussian):
                gaussian.post_processing()
                raise RuntimeError("fixture")
        self.assertEqual(supports, [])
        self.assertIs(gaussian.post_processing, original)

    def test_cli_dry_run_creates_nothing_and_does_not_require_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream = Path(temporary) / "upstream"
            (upstream / "config").mkdir(parents=True)
            (upstream / "config/main.yaml").write_text("fixture: true")
            run = Path(temporary) / "new-run"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/activegs/run_benchmark.py"),
                                     "--upstream", str(upstream), "--run-dir", str(run),
                                     "--methods", "defect", "refine_only", "--seeds", "0",
                                     "--frames", "6", "--prefix-frames", "2", "--dry-run"],
                                    capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["protocol"]["checkpoints"], [2, 4, 6])
            self.assertEqual(plan["status"], "plan_only_no_gpu_allocation")
            self.assertFalse(run.exists())


if __name__ == "__main__":
    unittest.main()
