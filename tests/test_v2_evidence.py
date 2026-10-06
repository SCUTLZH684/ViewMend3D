"""CPU mutation checks of provenance and measured v2 evidence, not reconstruction results."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/activegs"))
from viewmend3d.protocol import Protocol, optimization_recipe, validate_recipe_record, PROTOCOL_VERSION
from viewmend3d.campaign_profiles import campaign_spec_sha256, get_profile
from viewmend3d.diagnostic_audit import audit_guarded_event
from aggregate_benchmark import aggregate, read_run
import analyze_v2_campaign as analysis
import test_aggregate as fixtures


def diagnostic(method="defect_guarded", event=21, constant=False):
    spec = optimization_recipe("optimization-v2")
    baseline = [.2625, .2375] + [None] * 98
    defect = [.1, .1] if constant else [0., .1]
    bonus = [0., 0.] if constant else [0., .05]
    selected = 0 if constant else 1
    return {"method": method, "step": event, "status": "selected", "scoring_version": "bounded_geometry_v2",
            "geometry_backend": "batched", "geometry_measured": True, "geometry_measured_count": 100,
            "geometry_batch_count": 1, "normalization_domain": "reachable_candidates",
            "geometry_normalization": "peak_reachable_candidates", "geometry_thresholds": spec["geometry_thresholds"],
            "use_depth_gate": spec["methods"][method]["use_depth_gate"],
            "depth_discontinuity_gate": spec["methods"][method]["use_depth_gate"],
            "path_length_factor": .5, "render_ratio": .25, "render_resolution": [128, 128], "explore_weight": 1000.,
            "shared_scene_bbox": True, "geometry_weight": 0., "future_candidate_observation_queries": 0,
            "all_zero_utility_fallback": False,
            "planning_mask_source": "current_map", "base": [55., 45.] + [0.] * 98,
            "defect": defect + [100.] * 98, "path_lengths": [57.5, 42.5] + [-1.] * 98,
            "reachable": [True, True] + [False] * 98, "baseline_scores": baseline,
            "geometry_bonus": bonus + [0.] * 98,
            "defect_peak_normalized": ([0., 0.] if constant else [0., 1.]) + [0.] * 98,
            "final_scores": [baseline[i] + bonus[i] for i in range(2)] + [None] * 98,
            "geometry_beta": .1, "geometry_bonus_cap": .05, "n_reachable": 2, "geometry_peak": .1,
            "geometry_range": 0. if constant else .1, "geometry_signal_active": True,
            "geometry_discriminative": not constant, "geometry_active": not constant,
            "geometry_fallback_reason": "constant_geometry" if constant else "none",
            "baseline_selected_index": 0, "selected_index": selected, "selected_baseline_score": baseline[selected],
            "baseline_regret": baseline[0] - baseline[selected], "regret_tolerance": 1e-7,
            "regret_bound_satisfied": True, "selection_changed": not constant, "utility_seconds": .01}


class V2EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = fixtures.EvidenceAuditTests()
        self.fixtures.setUp()
        self.addCleanup(self.fixtures.tearDown)

    def v2_fixture(self, method="defect_guarded", seed=0):
        path = self.fixtures.fixture(method, seed)
        record = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
        recipe = Protocol(recipe="optimization-v2", campaign_spec_sha256=campaign_spec_sha256("optimization-v2")).validate().as_dict()
        record.update(version=recipe["version"], protocol=recipe, scene_assets_sha256="c" * 64,
                      scoring_recipe=recipe["recipe_spec"]["methods"][method])
        record.update({key: recipe[key] for key in ("recipe", "recipe_sha256", "campaign_spec_sha256")})
        (path / "protocol.json").write_text(json.dumps(record), encoding="utf-8")
        config = {"planner": {"sample_num": 100, "max_roi_sample_num": 30, "path_length_factor": .5,
                              "render_ratio": .25, "explore_weight": 1000., "geometry_weight": .5,
                              "geometry_beta": .1, "geometry_backend": "batched", "planner_name": method,
                              "geometry_thresholds": recipe["recipe_spec"]["geometry_thresholds"]},
                  "mapper": {"gaussian_map": {"optimization_steps": 10}}}
        (path / "exp_config.json").write_text(json.dumps(config), encoding="utf-8")
        if method.startswith("defect_guarded"):
            folder = path / "diagnostics"
            folder.mkdir()
            for event in range(21, 61):
                (folder / f"step_{event:04d}.json").write_text(json.dumps(diagnostic(method, event)), encoding="utf-8")
        return path

    def test_v1_default_schema_and_rng_version_are_preserved(self):
        legacy = Protocol().as_dict()
        self.assertEqual(legacy["version"], PROTOCOL_VERSION)
        self.assertFalse({"recipe", "recipe_spec", "recipe_sha256", "campaign_spec_sha256"} & set(legacy))
        validate_recipe_record(legacy)

    def test_complete_recipe_and_campaign_hash_cannot_be_changed(self):
        with self.assertRaisesRegex(ValueError, "specification"):
            Protocol(recipe="optimization-v2").validate()
        record = Protocol(recipe="optimization-v2", campaign_spec_sha256=campaign_spec_sha256("optimization-v2")).validate().as_dict()
        for mutation in (lambda value: value.pop("recipe_spec"),
                         lambda value: value["recipe_spec"]["methods"]["defect_guarded"].update(geometry_beta=.2),
                         lambda value: value.update(campaign_spec_sha256="0" * 64)):
            changed = copy.deepcopy(record)
            mutation(changed)
            with self.assertRaises(ValueError):
                validate_recipe_record(changed)

    def test_v2_cli_dry_run_is_versioned_and_rejects_different_scene(self):
        upstream = self.fixtures.root / "upstream"
        (upstream / "config").mkdir(parents=True)
        (upstream / "config/main.yaml").write_text("fixture: true")
        run = self.fixtures.root / "new-run"
        argv = [sys.executable, str(ROOT / "scripts/activegs/run_benchmark.py"),
                "--upstream", str(upstream), "--run-dir", str(run), "--methods", "defect_guarded",
                "--seeds", "0", "--frames", "6", "--prefix-frames", "2", "--recipe", "optimization-v2",
                "--campaign-spec-sha256", campaign_spec_sha256("optimization-v2"), "--dry-run"]
        planned = subprocess.run(argv, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(planned.stdout)["protocol"]["version"], "viewmend-observed-only-v2")
        rejected = subprocess.run(argv + ["--scene", "replica/room0"], capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("fixes scene", rejected.stderr)
        self.assertFalse(run.exists())

    def test_audits_actual_bonus_selection_and_unreachable_domain(self):
        row = audit_guarded_event(diagnostic(), "defect_guarded", 21)
        self.assertTrue(row["selection_changed"])
        self.assertLess(row["baseline_regret"], row["bonus_cap"])
        for key, value in (("geometry_peak", 100.), ("selected_index", 0), ("n_reachable", 100),
                           ("regret_bound_satisfied", False), ("regret_tolerance", .1)):
            changed = diagnostic()
            changed[key] = value
            with self.assertRaises(ValueError):
                audit_guarded_event(changed, "defect_guarded", 21)

    def test_inactive_reason_timing_and_recorded_parameters_are_recomputed(self):
        row = audit_guarded_event(diagnostic(constant=True), "defect_guarded", 21)
        self.assertEqual(row["fallback_reason"], "constant_geometry")
        for key, value in (("geometry_fallback_reason", "base_all_zero"), ("utility_seconds", -.1),
                           ("path_length_factor", 5.), ("render_ratio", .5), ("explore_weight", 50),
                           ("render_resolution", [64, 64]), ("future_candidate_observation_queries", 1)):
            changed = diagnostic(constant=True)
            changed[key] = value
            with self.assertRaises(ValueError):
                audit_guarded_event(changed, "defect_guarded", 21)

    def test_zero_base_fallback_cannot_record_impossible_random_scores(self):
        data = diagnostic(constant=True)
        data.update(base=[0.] * 100, all_zero_utility_fallback=True,
                    geometry_fallback_reason="base_all_zero", baseline_scores=[.1, .9] + [None] * 98,
                    final_scores=[.1, .9] + [None] * 98, selected_index=1, baseline_selected_index=1,
                    selected_baseline_score=.9, baseline_regret=0.)
        audit_guarded_event(data, "defect_guarded", 21)
        for invalid in (-1., 1.):
            changed = copy.deepcopy(data)
            changed.update(baseline_scores=[invalid] * 2 + [None] * 98,
                           final_scores=[invalid] * 2 + [None] * 98, selected_index=0,
                           baseline_selected_index=0, selected_baseline_score=invalid)
            with self.assertRaisesRegex(ValueError, "random range"):
                audit_guarded_event(changed, "defect_guarded", 21)

    def test_aggregate_requires_versioned_scoring_and_full_diagnostics(self):
        guarded = self.v2_fixture()
        confidence = self.v2_fixture("confidence_nooracle")
        report = aggregate([confidence, guarded], ["confidence_nooracle", "defect_guarded"], [0])
        self.assertEqual(report["version"], "viewmend-observed-only-v2")
        self.assertEqual(report["guarded_diagnostics"][0]["validated_events"], 40)
        (guarded / "diagnostics/step_0021.json").unlink()
        with self.assertRaisesRegex(ValueError, "diagnostic chain"):
            read_run(guarded)

    def test_missing_configuration_or_changed_method_recipe_is_rejected(self):
        path = self.v2_fixture()
        self.fixtures.mutate(path, "exp_config.json", lambda value: value["planner"].update(geometry_beta=.2))
        with self.assertRaisesRegex(ValueError, "planner configuration"):
            read_run(path)
        self.fixtures.mutate(path, "exp_config.json", lambda value: value["planner"].update(geometry_beta=.1))
        self.fixtures.mutate(path, "protocol.json", lambda value: value["scoring_recipe"].update(use_depth_gate=False))
        with self.assertRaisesRegex(ValueError, "method recipe"):
            read_run(path)

    def test_partial_campaign_never_writes_quality_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                status = analysis.main(["--root", folder, "--output", str(output)])
            self.assertEqual(status, 2)
            self.assertFalse(output.exists())
            self.assertFalse(json.loads(captured.getvalue())["quality_summary_generated"])

    def test_partitions_are_fixed_and_cannot_be_pooled_as_one_pair(self):
        stages = get_profile("optimization-v2")["stages"]
        self.assertEqual([len(stage["methods"]) * len(stage["seeds"]) for stage in stages], [3, 8, 4, 6, 6])
        runs = [{"method": method, "seed": seed, "metrics": {"completion_cm": seed + adjustment}}
                for method, adjustment in (("confidence_nooracle", 0), ("defect_guarded", .1)) for seed in (0, 1)]
        report = analysis.paired_statistics(runs, "defect_guarded", "confidence_nooracle", [0, 1], {"metrics": ["completion_cm"]})
        self.assertEqual(report["metrics"]["completion_cm"]["n"], 2)
        with self.assertRaisesRegex(ValueError, "declared seed"):
            analysis.paired_statistics(runs, "defect_guarded", "confidence_nooracle", [0, 1, 3], {"metrics": ["completion_cm"]})

    def test_export_audit_verifies_actual_camera_orientation_and_intrinsics(self):
        path = self.v2_fixture()
        protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
        result = json.loads((path / "final_result.json").read_text(encoding="utf-8"))
        # Fixture uses actual valid camera pickle and mesh payloads, not a fake
        # success report. Mock only the export JSON read to test mutations.
        from check_artifacts import read_cameras
        cameras = read_cameras(path / "map/cameras_060.pkl")
        exported = [{"position": row[3:12:4], "rotation": [row[r * 4 + c] for r in range(3) for c in range(3)],
                     "intrinsic": row[16:]} for row in cameras]
        manifest = {"status": "completed", "method_id": "defect_guarded", "seed": 0,
                    "protocol": protocol, "camera_convention": "OpenCV camera-to-world", "cameras": exported,
                    "trajectory": [camera["position"] for camera in exported], "path_stops": list(range(1, len(cameras) + 1)),
                    "checkpoints": []}
        run = self.fixtures.root / ("heldout_observations-" + "a" * 32)
        folder = self.fixtures.root / ("runs/web-assets/campaign-optimization-v2-heldout_observations-" + "a" * 32 + "-defect_guarded-s0")
        folder.mkdir(parents=True)
        import shutil
        from check_artifacts import ply_counts
        for index, event in enumerate(result["step"]):
            name = f"mesh_{event:03d}.ply"
            shutil.copyfile(path / "map" / name, folder / name)
            counts = ply_counts(folder / name)
            manifest["checkpoints"].append({"step": event, "update_event": event,
                "observation_count": result["observation_count"][index], "camera_count": result["observation_count"][index],
                "mesh": name, "preview": {"vertices": counts["vertex"], "faces": counts["face"]},
                **{key: result[source][index] * factor for key, (source, factor) in analysis.METRICS.items()}})
        stage = {"name": "heldout_observations"}
        with patch.object(analysis, "read", return_value=manifest):
            analysis.audit_export(self.fixtures.root, run, stage, path, protocol, result, {})
        for field, value in (("rotation", [0.] * 9), ("intrinsic", [999.] * 9)):
            changed = copy.deepcopy(manifest)
            changed["cameras"][0][field] = value
            with patch.object(analysis, "read", return_value=changed):
                with self.assertRaisesRegex(ValueError, "calibration"):
                    analysis.audit_export(self.fixtures.root, run, stage, path, protocol, result, {})


if __name__ == "__main__":
    unittest.main()
