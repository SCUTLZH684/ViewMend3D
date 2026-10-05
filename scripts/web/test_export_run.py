"""CPU checks for experiment identity and acquisition/update accounting."""
import json
import tempfile
import unittest
from pathlib import Path

import export_run


class ExportSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name).resolve()
        self.experiment = self.run / "experiments/benchmark/replica/office0/refine_only/2"
        self.experiment.mkdir(parents=True)
        (self.experiment / "final_result.json").write_text("{}")
        self.protocol = {"version": "v1", "method": "refine_only", "seed": 2,
                         "protocol": {"mode": "observations", "observations": 60, "prefix": 20},
                         "prefix_sha256": "identical-checkpoint", "eval_seed": 123}
        self.write_protocol(self.protocol)

    def tearDown(self):
        self.temp.cleanup()

    def write_protocol(self, value):
        (self.experiment / "protocol.json").write_text(json.dumps(value), encoding="utf-8")

    def test_refinement_events_do_not_create_observations(self):
        metrics = {"step": [20, 40, 60], "update_event": [20, 40, 60], "observation_count": [20, 20, 20]}
        self.assertEqual(export_run.count_for_checkpoint(metrics, 2, 20), (20, 60))
        with self.assertRaisesRegex(ValueError, "acquisition cameras"):
            export_run.count_for_checkpoint(metrics, 2, 60)

    def test_protocol_identity_and_pairing_are_recorded(self):
        metadata = export_run.experiment_metadata(self.run, self.experiment)
        self.assertEqual(metadata["method_id"], "refine_only")
        self.assertEqual(metadata["seed"], 2)
        self.assertEqual(metadata["protocol"], self.protocol)
        self.protocol["method"] = "defect"
        self.write_protocol(self.protocol)
        same_prefix = export_run.experiment_metadata(self.run, self.experiment)
        self.assertEqual(metadata["comparison_id"], same_prefix["comparison_id"])
        self.protocol["prefix_sha256"] = "different-prefix"
        self.write_protocol(self.protocol)
        self.assertNotEqual(metadata["comparison_id"], export_run.experiment_metadata(self.run, self.experiment)["comparison_id"])

    def test_original_directory_number_is_not_a_seed(self):
        original = self.run / "experiments/original/replica/office0/confidence/0"
        original.mkdir(parents=True)
        metadata = export_run.experiment_metadata(self.run, original)
        self.assertIsNone(metadata["seed"])
        self.assertTrue(metadata["protocol"]["candidate_oracle_mask"])

    def test_comparison_identity_includes_scene_and_configuration(self):
        metadata = export_run.experiment_metadata(self.run, self.experiment)
        for key, value in (("scene", "office1"), ("scene_mesh_sha256", "a" * 64),
                           ("scene_assets_sha256", "c" * 64),
                           ("context_hash", "b" * 64)):
            with self.subTest(key=key):
                self.write_protocol({**self.protocol, key: value})
                self.assertNotEqual(metadata["comparison_id"], export_run.experiment_metadata(self.run, self.experiment)["comparison_id"])

    def test_author_float_checkpoint_ids_from_saved_real_evidence(self):
        evidence = Path(__file__).resolve().parents[2] / "docs/reproduction/evidence/office0-final-result.json"
        metrics = json.loads(evidence.read_text(encoding="utf-8"))
        for index, step in enumerate(metrics["step"]):
            self.assertEqual(export_run.count_for_checkpoint(metrics, index, int(step)), (int(step), int(step)))
        for step in (40.5, True, float("inf")):
            with self.subTest(step=step), self.assertRaises(ValueError):
                export_run.count_for_checkpoint({"step": [step]}, 0, 40)

    def test_benchmark_requires_protocol_and_unambiguous_path(self):
        (self.experiment / "protocol.json").unlink()
        with self.assertRaisesRegex(ValueError, "require protocol"):
            export_run.experiment_metadata(self.run, self.experiment)
        self.assertEqual(export_run.resolve_experiment(self.run), self.experiment)
        another = self.run / "experiments/benchmark/replica/office0/defect/2"
        another.mkdir(parents=True)
        (another / "final_result.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Specify --experiment"):
            export_run.resolve_experiment(self.run)
        with self.assertRaises(ValueError):
            export_run.resolve_experiment(self.run, self.run.parent)

    def test_trajectory_checks_positions_as_well_as_counts(self):
        cameras = [{"position": [1, 0, 0]}, {"position": [2, 0, 0]}]
        export_run.validate_motion_path([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [2, 3], cameras)
        with self.assertRaisesRegex(ValueError, "align"):
            export_run.validate_motion_path([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [1, 3], cameras)
        with self.assertRaisesRegex(ValueError, "match"):
            export_run.validate_motion_path([[0, 0, 0]], [1], cameras)

    def test_diagnostics_export_selected_real_signal_only(self):
        folder = self.experiment / "diagnostics"
        folder.mkdir()
        output = self.run / "assets"
        output.mkdir()
        self.assertIsNone(export_run.export_diagnostics(self.experiment, output, 60))
        diagnostic = {"selected_index": 1, "geometry_active": True,
                      "candidates": [{"D": 0.0}, {"D": 0.25}], "heatmap": "selected.png"}
        (folder / "step_060.json").write_text(json.dumps(diagnostic))
        (folder / "selected.png").write_bytes(b"actual-test-pixels")
        exported = export_run.export_diagnostics(self.experiment, output, 60)
        self.assertEqual(exported["candidate"], {"D": 0.25})
        self.assertEqual((output / exported["heatmap"]).read_bytes(), b"actual-test-pixels")

    def test_planner_array_schema_and_shared_prefix_diagnostic(self):
        folder = self.run / "common-prefix/seed_2/diagnostics"
        folder.mkdir(parents=True)
        output = self.run / "assets"
        output.mkdir()
        diagnostic = {"selected_index": 1, "candidate_poses": [[0], [1]],
                      "exploration": [0.0, 0.4], "defect": [0.0, 0.02],
                      "final_scores": [0.0, 0.5], "valid_fraction": [0.1, 0.9],
                      "selected_heatmap": "step_0020_selected_heatmap.png"}
        (folder / "step_0020.json").write_text(json.dumps(diagnostic))
        (folder / "step_0020_selected_heatmap.png").write_bytes(b"prefix-heatmap")
        exported = export_run.export_diagnostics(self.experiment, output, 20, self.run)
        self.assertEqual(exported["candidate"]["D"], 0.02)
        self.assertEqual(exported["candidate"]["final_score"], 0.5)
        self.assertEqual(exported["candidate_count"], 2)
        self.assertIsNone(export_run.export_diagnostics(self.experiment, output, 40, self.run))


if __name__ == "__main__":
    unittest.main()
