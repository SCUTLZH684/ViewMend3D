"""Boundary cases for the frozen observed-only scoring protocol (CPU sufficient)."""

import math
import importlib.util
from pathlib import Path
import tempfile
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viewmend3d.scoring import (GeometryThresholds, combine_utilities,
                               geometry_defect, normalize_sum, view_scores)


def plane(size=16):
    normal = torch.zeros(3, size, size)
    normal[2] = 1.0
    return {"depth": torch.full((size, size), 2.0),
            "normal": normal, "depth_normal": normal.clone(),
            "opacity": torch.ones(size, size),
            "confidence": torch.full((size, size), 0.2),
            "depth_range": (0.1, 10.0)}


class GeometryTests(unittest.TestCase):
    def test_consistent_plane_and_border(self):
        result = geometry_defect(**plane())
        self.assertEqual(float(result["score"]), 0.0)
        self.assertAlmostEqual(float(result["valid_fraction"]), 14 * 14 / 256)
        self.assertFalse(bool(result["mask"][0].any()))
        self.assertTrue(bool(result["mask"][1:-1, 1:-1].all()))

    def test_directed_mismatch_has_image_area_denominator(self):
        data = plane()
        data["depth_normal"] *= -1
        result = geometry_defect(**data)
        self.assertAlmostEqual(float(result["score"]), 0.8 * 196 / 256, places=6)
        self.assertEqual(float(result["residual"][8, 8]), 1.0)
        data["confidence"].fill_(1.0)
        self.assertEqual(float(geometry_defect(**data)["score"]), 0.0)

    def test_residual_deadzone(self):
        data = plane()
        angle = math.radians(10)
        data["depth_normal"][0] = math.sin(angle)
        data["depth_normal"][2] = math.cos(angle)
        self.assertEqual(float(geometry_defect(**data)["score"]), 0.0)

    def test_depth_discontinuity_ablation_only_removes_depth_gate(self):
        data = plane()
        data["depth"][:, 8:] = 5.0
        data["depth_normal"][:, :, 7:9] *= -1
        gated = geometry_defect(**data)
        ungated = geometry_defect(**data, use_depth_gate=False)
        self.assertEqual(float(gated["score"]), 0.0)
        self.assertGreater(float(ungated["score"]), 0.0)
        self.assertFalse(bool(ungated["mask"][0].any()))

    def test_invalid_values_are_excluded_with_neighbors(self):
        for field, value in (("depth", -1.0), ("depth", float("nan")),
                             ("depth", float("inf")), ("depth", 11.0),
                             ("opacity", 0.69), ("confidence", float("nan"))):
            with self.subTest(field=field, value=value):
                data = plane()
                data["depth_normal"] *= -1
                data[field][8, 8] = value
                result = geometry_defect(**data)
                self.assertTrue(bool(torch.isfinite(result["heatmap"]).all()))
                self.assertFalse(bool(result["mask"][7:10, 7:10].any()))

    def test_degenerate_normals_and_minimum_area(self):
        data = plane()
        data["normal"].zero_()
        self.assertEqual(float(geometry_defect(**data)["score"]), 0.0)
        data = plane(size=4)
        data["depth_normal"] *= -1
        result = geometry_defect(**data, thresholds=GeometryThresholds(min_valid_fraction=0.3))
        self.assertAlmostEqual(float(result["valid_fraction"]), 0.25)
        self.assertFalse(bool(result["mask"].any()))
        self.assertEqual(float(result["score"]), 0.0)

    def test_batch_matches_individual_views(self):
        first, second = plane(), plane()
        second["depth_normal"] *= -1
        batch = {key: torch.stack((first[key], second[key])) for key in first if key != "depth_range"}
        result = geometry_defect(**batch, depth_range=first["depth_range"])
        self.assertEqual(tuple(result["heatmap"].shape), (2, 16, 16))
        for index, data in enumerate((first, second)):
            self.assertTrue(torch.allclose(result["score"][index], geometry_defect(**data)["score"]))

    def test_empty_geometry_and_invalid_thresholds_fail(self):
        with self.assertRaises(ValueError):
            geometry_defect(torch.empty(0, 0), torch.empty(3, 0, 0),
                            torch.empty(3, 0, 0), torch.empty(0, 0),
                            torch.empty(0, 0), (0.1, 10))
        with self.assertRaises(ValueError):
            GeometryThresholds(residual_deadzone=1.0)

    def test_depth_bounds_preserve_dtype_cast_and_validation(self):
        data = plane()
        data["depth_normal"] *= -1
        reference = geometry_defect(**data)
        for bounds in (torch.tensor([0.1, 10.0]), [[0.1, 10.0]],
                       torch.tensor([0.1, 10.0], dtype=torch.float64)):
            with self.subTest(bounds=bounds):
                actual = geometry_defect(**{**data, "depth_range": bounds})
                self.assertTrue(torch.equal(actual["mask"], reference["mask"]))
                self.assertTrue(torch.equal(actual["score"], reference["score"]))
        for bounds in ((1.0,), (1.0, 1.0), (2.0, 1.0),
                       (float("nan"), 10), (0.1, float("inf")),
                       (0.1, 3.5e38)):
            # The last range is finite as Python floats but overflows the
            # float32 image dtype; it must keep failing after host validation.
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                geometry_defect(**{**data, "depth_range": bounds})


class UtilityTests(unittest.TestCase):
    def test_lambda_zero_equivalence_including_unreachable(self):
        base = torch.tensor([20.0, 4.0, 3.0, 1000.0])
        defect = torch.tensor([0.0, 0.7, 0.2, 1.0])
        lengths = [2.0, 1.0, 3.0, float("inf")]
        expected = view_scores(base, lengths)
        actual = view_scores(combine_utilities(base, defect, weight=0.0), lengths)
        self.assertTrue(torch.allclose(actual[:3], expected[:3], atol=1e-7))
        self.assertEqual(int(actual.argmax()), int(expected.argmax()))
        self.assertTrue(bool(torch.isneginf(actual[-1])))

    def test_weak_and_zero_defect_do_not_amplify_noise(self):
        base = torch.tensor([2.0, 1.0])
        for defect in (torch.zeros(2), torch.tensor([1e-6, 2e-6])):
            self.assertTrue(torch.equal(combine_utilities(base, defect), normalize_sum(base)))
        active = combine_utilities(base, torch.tensor([0.0, 0.1]))
        self.assertTrue(torch.allclose(active, torch.tensor([2/3, 1/3 + 0.5])))

    def test_invalid_utilities_have_no_reward(self):
        utilities = torch.tensor([float("nan"), float("inf"), -1.0, 2.0])
        self.assertTrue(torch.equal(normalize_sum(utilities), torch.tensor([0., 0., 0., 1.])))

    def test_unreachable_and_zero_length_are_safe(self):
        scores = view_scores(torch.tensor([1.0, 100.0, 2.0]), [0.0, float("inf"), 0.0])
        self.assertTrue(torch.allclose(scores[[0, 2]], torch.tensor([1/3, 2/3])))
        self.assertEqual(int(scores.argmax()), 2)
        with self.assertRaisesRegex(RuntimeError, "no reachable"):
            view_scores(torch.ones(2), [float("inf"), float("nan")])
        with self.assertRaises(ValueError):
            view_scores(torch.empty(0), [])

    def test_zero_utility_fallback_is_reproducible_and_reachable(self):
        torch.manual_seed(19)
        first = view_scores(torch.zeros(4), [0.0, float("inf"), 2.0, -1.0])
        torch.manual_seed(19)
        second = view_scores(torch.zeros(4), [0.0, float("inf"), 2.0, -1.0])
        self.assertTrue(torch.equal(first, second))
        self.assertIn(int(first.argmax()), (0, 2))


class PlannerInterfaceTests(unittest.TestCase):
    """Exercise the real adapter with deterministic renderer/path test doubles.

    The pinned upstream imports CUDA rendering/evaluation dependencies; these
    interface tests intentionally import neither those modules nor a GPU.
    """

    def load_planner(self):
        # Keep Pillow's plugin registry outside the temporary sys.modules
        # patch; restoring that patch otherwise removes newly imported image
        # modules while retaining their referenced Image objects.
        Image.init()
        class Base:
            def __init__(self, cfg, device):
                self.device = device
                self.path_length_factor = cfg.path_length_factor
                self.pose = torch.eye(4)
                self.init = False
                self.flight_speed = 1.0

        class Renderer:
            calls = 0

            def __init__(self, *args, **kwargs):
                pass

            def render_view(self, index):
                Renderer.calls += 1
                data = plane(size=16)
                if index:
                    data["depth_normal"] *= -1
                return (torch.zeros(3, 16, 16), data["depth"].unsqueeze(0),
                        data["normal"], data["opacity"].unsqueeze(0),
                        data["depth_normal"], data["confidence"].unsqueeze(0),
                        None, None, None)

        modules = {name: ModuleType(name) for name in (
            "planning", "planning.plan_base", "planning.utils", "utils",
            "utils.common", "utils.operations")}
        modules["planning.plan_base"].PlanBase = Base
        modules["planning.utils"].cal_flight_time = lambda length, flight_speed: length / flight_speed
        modules["planning.utils"].wp2path = lambda *args: (_ for _ in ()).throw(AssertionError("zero-motion path should bypass upstream"))
        modules["utils.common"].Planner2Gui = lambda *args: None
        modules["utils.operations"].GaussianRenderer = Renderer
        module_path = Path(__file__).resolve().parents[1] / "src/viewmend3d/planner.py"
        specification = importlib.util.spec_from_file_location("viewmend3d._planner_test", module_path)
        module = importlib.util.module_from_spec(specification)
        with patch.dict("sys.modules", modules):
            specification.loader.exec_module(module)
        return module.DefectPlanner, Renderer

    def fixtures(self):
        cfg = SimpleNamespace(path_length_factor=0.5, render_ratio=0.25,
                              explore_weight=1000.0)
        class Simulator:
            resolution = (64, 64)
            depth_range = (0.1, 10.0)
            intrinsic = torch.eye(3)
            has_missing_surface = True

            def simulate(self, *args, **kwargs):
                raise AssertionError("future candidate observation was queried")

        gaussian = SimpleNamespace(get_attr=lambda: None, background_color=torch.zeros(4),
                                   scene_near=0.001, scene_far=10.0)
        voxel = SimpleNamespace(voxel_centers=torch.zeros(4, 3),
                                unexplored_mask=torch.ones(4, dtype=torch.bool),
                                cal_visible_mask=lambda *args: torch.tensor([True, False, True, False]))
        candidates = torch.eye(4).repeat(2, 1, 1)
        candidates[1, 0, 3] = 0.1
        return cfg, Simulator(), gaussian, voxel, candidates

    def test_no_oracle_single_render_per_candidate_and_lambda_zero(self):
        Planner, Renderer = self.load_planner()
        cfg, simulator, gaussian, voxel, candidates = self.fixtures()
        baseline = Planner(cfg, "cpu", method="confidence_nooracle")
        baseline_utility, _ = baseline.cal_utility(gaussian, voxel, candidates, simulator)
        baseline_scores = baseline.cal_view_scores(baseline_utility, [1.0, 1.0])
        self.assertEqual(Renderer.calls, 2)
        defect = Planner(cfg, "cpu", method="defect", geometry_weight=0)
        defect_utility, _ = defect.cal_utility(gaussian, voxel, candidates, simulator)
        defect_scores = defect.cal_view_scores(defect_utility, [1.0, 1.0])
        self.assertEqual(Renderer.calls, 4)
        self.assertTrue(torch.equal(baseline_scores, defect_scores))
        self.assertEqual(defect.last_diagnostics["future_candidate_observation_queries"], 0)
        self.assertGreater(defect.last_diagnostics["defect"][1], 0)

    def test_renderer_images_are_unchanged_during_utility_scoring(self):
        Planner, Renderer = self.load_planner()
        cfg, simulator, gaussian, voxel, candidates = self.fixtures()
        data = plane()
        data["depth"][3, 3] = float("nan")
        data["depth"][4, 4] = 0.0
        data["depth"][5, 5] = 12.0
        data["confidence"][6, 6] = float("nan")
        data["depth_normal"] *= -1
        images = (torch.zeros(3, 16, 16), data["depth"].unsqueeze(0),
                  data["normal"], data["opacity"].unsqueeze(0),
                  data["depth_normal"], data["confidence"].unsqueeze(0))
        original = [value.clone() for value in images]
        with patch.object(Renderer, "render_view", return_value=(*images, None, None, None)):
            planner = Planner(cfg, "cpu")
            utility, _ = planner.cal_utility(gaussian, voxel, candidates, simulator)
        self.assertTrue(bool(torch.isfinite(utility).all()))
        for actual, expected in zip(images, original):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    def test_reachable_normalization_and_diagnostics(self):
        Planner, _ = self.load_planner()
        cfg, simulator, gaussian, voxel, candidates = self.fixtures()
        with tempfile.TemporaryDirectory() as directory:
            planner = Planner(cfg, "cpu", diagnostics_dir=directory)
            utility, _ = planner.cal_utility(gaussian, voxel, candidates, simulator)
            scores = planner.cal_view_scores(utility, [1.0, float("inf")])
            self.assertEqual(int(scores.argmax()), 0)
            self.assertFalse(planner.last_diagnostics["geometry_active"])
            self.assertIsNone(planner.last_diagnostics["final_scores"][1])
            planner._write_diagnostics()
            self.assertTrue((Path(directory) / "step_0000.json").is_file())
            self.assertTrue((Path(directory) / "step_0000_selected_heatmap.png").is_file())

    def test_zero_motion_returns_observation_pose(self):
        Planner, _ = self.load_planner()
        cfg, simulator, _, _, _ = self.fixtures()
        planner = Planner(cfg, "cpu")
        voxel = SimpleNamespace(xyz_2_index=lambda _: (0, 0, 0),
                                index_2_xyz=lambda _: torch.zeros(1, 3))
        path = planner.plan((None, voxel), simulator, None)
        self.assertEqual(tuple(path.shape), (1, 4, 4))
        self.assertEqual(planner.last_diagnostics["travel_distance"], 0.0)

    def test_small_rotation_path_reaches_the_selected_camera_pose(self):
        Planner, _ = self.load_planner()
        cfg, simulator, _, _, _ = self.fixtures()
        planner = Planner(cfg, "cpu", method="random_matched")
        start = planner.pose.clone()
        goal = start.clone()
        angle = 0.05
        goal[:3, :3] = torch.tensor([[math.cos(angle), 0, math.sin(angle)],
                                    [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        planner.init = True
        planner.sample_num, planner.max_roi_sample_num = 1, 0
        planner.q_planner2gui = SimpleNamespace(put=lambda _: None)
        planner.get_robot_space = lambda _: None
        planner.generate_random_candidates = lambda *args: goal.unsqueeze(0)
        planner.path_planner = SimpleNamespace(search_goal=lambda *args: ([[[0, 0, 0]]], [0.0]))
        voxel = SimpleNamespace(update_graph=lambda _: None, index_2_xyz=lambda _: torch.zeros(1, 3))
        # This is the observed output of the author's one-sample small-turn
        # helper. The adapter must correct the endpoint before sensor capture.
        with patch.dict(Planner.plan.__globals__, {"wp2path": lambda *args: (start.unsqueeze(0), 0.0)}):
            path = planner.plan((None, voxel), simulator, None)
        self.assertTrue(torch.equal(path[-1], goal))
        self.assertTrue(torch.equal(planner.pose, path[-1]))


if __name__ == "__main__":
    unittest.main()
