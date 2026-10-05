"""Evidence-audit regression tests; fixtures are not reconstruction results."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/activegs"))
from aggregate_benchmark import aggregate, read_run
from check_artifacts import check
from test_artifacts import write_output_chain


class EvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, method, seed):
        path = self.root / method / str(seed)
        path.mkdir(parents=True)
        protocol = {"version": "viewmend-observed-only-v1", "method": method, "seed": seed,
                    "protocol": {"mode": "observations", "observations": 60, "prefix": 20,
                                 "future_candidate_depth_mask": False,
                                 "mapping_optimizer_steps_per_event": 10, "safety_event_cap": 10000},
                    "prefix_sha256": str(seed) * 64, "prefix_camera_sha256": "e" * 64,
                    "eval_seed": 10, "source_versions": {"fixture_only": True},
                    "scene": "office0", "scene_mesh_sha256": "a" * 64, "context_hash": "b" * 64,
                    "cost": {"mission_seconds": 10, "wall_seconds": 11, "planning_seconds": 1,
                             "mapping_seconds": 8, "sensor_seconds": 1, "path_length_m": 1,
                             "observations": 60, "optimizer_steps": 600, "events": 60,
                             "simulated_flight_seconds": 1, "branch_observation_calls": 40,
                             "blocked_future_observation_calls": 0, "checkpoint_count": 3,
                             "stop_reason": "event_limit", "method": method, "seed": seed}}
        result = {"step": [20, 40, 60], "update_event": [20, 40, 60], "observation_count": [20, 40, 60],
                  "time": [3, 6, 10], "path_length": [0, 1, 1],
                  "mesh_accuracy": [1, 1, 1], "mesh_completion": [2, 2, 2],
                  "mesh_completion_ratio": [70, 80, 90], "mesh_chamfer_distance": [0.015] * 3,
                  "evaluation": {"fixture_only": True}}
        steps = [{"event": event, "observations": event,
                  "mission_seconds": event * .15 if event <= 40 else 6 + (event - 40) * .2,
                  "path_length_m": 0 if event <= 20 else min((event - 20) / 20, 1),
                  "planning_seconds": 1 / 60, "mapping_seconds": 8 / 60, "sensor_seconds": 1 / 60}
                 for event in range(1, 61)]
        checkpoints = [{"event": event, "observations": event,
                        "mission_seconds": result["time"][i], "path_length_m": result["path_length"][i]}
                       for i, event in enumerate(result["step"])]
        write_output_chain(path, result)
        for name, value in (("protocol.json", protocol), ("artifact-check.json", check(path)),
                            ("steps.json", steps), ("checkpoints.json", checkpoints),
                            ("run-summary.json", protocol["cost"])):
            (path / name).write_text(json.dumps(value), encoding="utf-8")
        return path

    def mutate(self, path, filename, change):
        target = path / filename
        value = json.loads(target.read_text(encoding="utf-8"))
        change(value)
        target.write_text(json.dumps(value), encoding="utf-8")

    def test_requires_every_requested_pair(self):
        paths = [self.fixture(method, seed) for method in ("confidence_nooracle", "defect") for seed in (0, 1, 2)]
        report = aggregate(paths, ["confidence_nooracle", "defect"], [0, 1, 2])
        self.assertEqual(report["methods"]["defect"]["mesh_completion"]["n"], 3)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            aggregate(paths[:-1], ["confidence_nooracle", "defect"], [0, 1, 2])

    def test_duplicate_requested_seeds_cannot_inflate_sample_size(self):
        paths = [self.fixture("defect", 0)]
        with self.assertRaisesRegex(ValueError, "valid and unique"):
            aggregate(paths, ["defect"], [0, 0])
        with self.assertRaisesRegex(ValueError, "valid and unique"):
            aggregate(paths, ["defect", "defect"], [0])

    def test_rejects_false_pairing_even_when_all_seeds_exist(self):
        paths = [self.fixture(method, 0) for method in ("confidence_nooracle", "defect")]
        self.mutate(paths[1], "protocol.json", lambda value: value.update(prefix_sha256="a" * 64))
        with self.assertRaisesRegex(ValueError, "Unequal actual prefixes"):
            aggregate(paths, ["confidence_nooracle", "defect"], [0])

    def test_rejects_oracle_and_mismatched_metric_units(self):
        path = self.fixture("defect", 0)
        self.mutate(path, "protocol.json", lambda value: value["protocol"].update(future_candidate_depth_mask=True))
        with self.assertRaisesRegex(ValueError, "Future candidate"):
            read_run(path)
        self.mutate(path, "protocol.json", lambda value: value["protocol"].update(future_candidate_depth_mask=False))
        self.mutate(path, "final_result.json", lambda value: value.update(mesh_chamfer_distance=[15, 15, 15]))
        with self.assertRaisesRegex(ValueError, "units"):
            read_run(path)

    def test_refinement_update_events_are_not_new_observations(self):
        path = self.fixture("refine_only", 0)
        with self.assertRaisesRegex(ValueError, "must not acquire"):
            read_run(path)
        self.mutate(path, "final_result.json", lambda value: value.update(observation_count=[20, 20, 20]))
        result = json.loads((path / "final_result.json").read_text(encoding="utf-8"))
        result["path_length"] = [0, 0, 0]
        write_output_chain(path, result)
        (path / "artifact-check.json").write_text(json.dumps(check(path)), encoding="utf-8")
        steps = json.loads((path / "steps.json").read_text(encoding="utf-8"))
        for step in steps:
            step.update(observations=min(step["event"], 20), path_length_m=0)
            if step["event"] > 20:
                step.update(planning_seconds=0, sensor_seconds=0,
                            mapping_seconds=.15 if step["event"] <= 40 else .2)
        (path / "steps.json").write_text(json.dumps(steps), encoding="utf-8")
        self.mutate(path, "checkpoints.json", lambda value: [checkpoint.update(observations=20, path_length_m=0) for checkpoint in value])
        self.mutate(path, "protocol.json", lambda value: value["cost"].update(
            observations=20, branch_observation_calls=0, path_length_m=0,
            planning_seconds=1 / 3, sensor_seconds=1 / 3, mapping_seconds=29 / 3, simulated_flight_seconds=0))
        protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
        (path / "run-summary.json").write_text(json.dumps(protocol["cost"]), encoding="utf-8")
        self.assertEqual(read_run(path)["result"]["update_event"][-1], 60)

    def test_rejects_cross_seed_scene_or_configuration_changes(self):
        paths = [self.fixture(method, seed) for method in ("confidence_nooracle", "defect") for seed in (0, 1)]
        for key, different, original in (("scene", "office1", "office0"),
                                         ("scene_mesh_sha256", "c" * 64, "a" * 64),
                                         ("context_hash", "d" * 64, "b" * 64)):
            with self.subTest(key=key):
                for path in (paths[1], paths[3]):
                    self.mutate(path, "protocol.json", lambda value: value.update({key: different}))
                with self.assertRaisesRegex(ValueError, "Mixed scenes"):
                    aggregate(paths, ["confidence_nooracle", "defect"], [0, 1])
                for path in (paths[1], paths[3]):
                    self.mutate(path, "protocol.json", lambda value: value.update({key: original}))

    def test_requires_successful_complete_artifact_validation(self):
        path = self.fixture("defect", 0)
        for changes in ({"artifact_chain_passed": False}, {"checkpoint_count": 2},
                        {"observation_count": [20, 40, 59]}, {"artifacts": [{"checkpoint": 20}]}):
            original = (path / "artifact-check.json").read_text(encoding="utf-8")
            with self.subTest(changes=changes):
                self.mutate(path, "artifact-check.json", lambda value: value.update(changes))
                with self.assertRaisesRegex(ValueError, "Artifact validation"):
                    read_run(path)
                (path / "artifact-check.json").write_text(original, encoding="utf-8")
        (path / "artifact-check.json").unlink()
        with self.assertRaisesRegex(ValueError, "Completed artifact validation"):
            read_run(path)

    def test_saved_success_report_cannot_hide_missing_or_replaced_files(self):
        path = self.fixture("defect", 0)
        mesh = path / "map/mesh_060.ply"
        original = mesh.read_bytes()
        mesh.unlink()
        with self.assertRaisesRegex(ValueError, "Missing/empty artifact"):
            read_run(path)
        mesh.write_bytes(original.replace(b"1 0 0", b"2 0 0"))
        with self.assertRaisesRegex(ValueError, "fingerprints are stale"):
            read_run(path)

    def test_costs_must_match_actual_last_checkpoint_and_event_ledger(self):
        path = self.fixture("defect", 0)
        self.mutate(path, "protocol.json", lambda value: value["cost"].update(mission_seconds=9))
        self.mutate(path, "run-summary.json", lambda value: value.update(mission_seconds=9))
        with self.assertRaisesRegex(ValueError, "mission_seconds"):
            read_run(path)

    def test_time_protocol_cannot_finish_by_safety_cap(self):
        path = self.fixture("defect", 0)
        self.mutate(path, "protocol.json", lambda value: value["protocol"].update(mode="time", seconds=11))
        self.mutate(path, "protocol.json", lambda value: value["cost"].update(stop_reason="safety_event_cap"))
        self.mutate(path, "run-summary.json", lambda value: value.update(stop_reason="safety_event_cap"))
        with self.assertRaisesRegex(ValueError, "actual time budget"):
            read_run(path)
        self.mutate(path, "protocol.json", lambda value: value["cost"].update(stop_reason="time_budget"))
        self.mutate(path, "run-summary.json", lambda value: value.update(stop_reason="time_budget"))
        with self.assertRaisesRegex(ValueError, "budget was not completed"):
            read_run(path)
        self.mutate(path, "protocol.json", lambda value: value["protocol"].update(seconds=9.9))
        self.assertEqual(read_run(path)["cost"]["stop_reason"], "time_budget")
        self.mutate(path, "protocol.json", lambda value: value["protocol"].update(seconds=9))
        with self.assertRaisesRegex(ValueError, "continued after exhausting"):
            read_run(path)

    def test_rejects_corrupt_event_and_checkpoint_ledgers(self):
        path = self.fixture("defect", 0)
        for filename, mutation, message in (
                ("steps.json", lambda value: value[30].update(event=30), "exactly once"),
                ("steps.json", lambda value: value[30].update(observations=30), "observation counts"),
                ("checkpoints.json", lambda value: value[-1].update(mission_seconds=8), "values disagree")):
            original = (path / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.mutate(path, filename, mutation)
                with self.assertRaisesRegex(ValueError, message):
                    read_run(path)
            (path / filename).write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
