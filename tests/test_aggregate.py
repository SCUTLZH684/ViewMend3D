"""Evidence-audit regression tests; fixtures are not reconstruction results."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/activegs"))
from aggregate_benchmark import aggregate, read_run


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
                                 "future_candidate_depth_mask": False},
                    "prefix_sha256": str(seed) * 64, "prefix_camera_sha256": "fixture",
                    "eval_seed": 10, "source_versions": {"fixture_only": True},
                    "scene": "office0", "scene_mesh_sha256": "a" * 64, "context_hash": "b" * 64,
                    "cost": {"mission_seconds": 10, "wall_seconds": 11, "planning_seconds": 1,
                             "mapping_seconds": 8, "sensor_seconds": 1, "path_length_m": 1,
                             "observations": 60, "optimizer_steps": 600}}
        result = {"step": [20, 40, 60], "update_event": [20, 40, 60], "observation_count": [20, 40, 60],
                  "time": [3, 6, 10], "path_length": [0, 1, 1],
                  "mesh_accuracy": [1, 1, 1], "mesh_completion": [2, 2, 2],
                  "mesh_completion_ratio": [70, 80, 90], "mesh_chamfer_distance": [0.015] * 3,
                  "evaluation": {"fixture_only": True}}
        check = {"artifact_chain_passed": True, "checkpoint_count": 3,
                 "artifacts": [{"checkpoint": event} for event in (20, 40, 60)],
                 "observation_count": [20, 40, 60]}
        for name, value in (("protocol.json", protocol), ("final_result.json", result), ("artifact-check.json", check)):
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
        with self.assertRaisesRegex(ValueError, "distance units"):
            read_run(path)

    def test_refinement_update_events_are_not_new_observations(self):
        path = self.fixture("refine_only", 0)
        with self.assertRaisesRegex(ValueError, "must not acquire"):
            read_run(path)
        self.mutate(path, "final_result.json", lambda value: value.update(observation_count=[20, 20, 20]))
        self.mutate(path, "artifact-check.json", lambda value: value.update(observation_count=[20, 20, 20]))
        self.mutate(path, "protocol.json", lambda value: value["cost"].update(observations=20))
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


if __name__ == "__main__":
    unittest.main()
