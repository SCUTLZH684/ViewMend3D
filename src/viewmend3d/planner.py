"""Observed-only NBV extension for the pinned ActiveGS implementation.

Add the upstream checkout to sys.path before importing this module. The
simulator supplies camera metadata only during planning; it is never queried
for a candidate observation. The benchmark runner acquires the chosen view.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from planning.plan_base import PlanBase
from planning.utils import cal_flight_time, wp2path
from utils.common import Planner2Gui
from utils.operations import GaussianRenderer

from .scoring import (GeometryThresholds, combine_utilities, geometry_defect,
                      nonnegative_finite, normalize_sum, view_scores)


METHODS = ("defect", "confidence_nooracle", "random_matched", "defect_no_gate")


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class DefectPlanner(PlanBase):
    """Common candidate recipe and safe path scoring for all matched methods.

    ``last_diagnostics`` is JSON-serializable and can be included in runner
    JSONL output. Set ``diagnostics_step`` to a runner event number if desired;
    otherwise plan-call indices are used. Artifact writing is optional.
    """

    def __init__(self, cfg, device, method="defect", diagnostics_dir=None,
                 geometry_weight=None, thresholds=None):
        super().__init__(cfg, torch.device(device))
        if method not in METHODS:
            raise ValueError(f"unknown planner method: {method}")
        self.method = method
        self.render_ratio = float(cfg.render_ratio)
        self.explore_weight = float(cfg.explore_weight)
        self.geometry_weight = float(geometry_weight if geometry_weight is not None
                                     else getattr(cfg, "geometry_weight", 0.5))
        if not 0 < self.render_ratio <= 1 or not math.isfinite(self.explore_weight) or self.explore_weight < 0:
            raise ValueError("invalid render_ratio or explore_weight")
        if not math.isfinite(self.geometry_weight) or self.geometry_weight < 0:
            raise ValueError("geometry_weight must be finite and nonnegative")
        if thresholds is None:
            thresholds = getattr(cfg, "geometry_thresholds", None)
        self.thresholds = (thresholds if isinstance(thresholds, GeometryThresholds)
                           else GeometryThresholds(**dict(thresholds or {})))
        self.diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir is not None else None
        self.diagnostics_step = None
        self.last_diagnostics = {}
        self._plan_calls = 0
        self._heatmaps = []
        self.diagnostics_io_seconds = 0.0

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _new_diagnostics(self):
        return {
            "step": int(self.diagnostics_step if self.diagnostics_step is not None
                        else self._plan_calls),
            "method": self.method,
            "future_candidate_observation_queries": 0,
            "planning_mask_source": "current_map",
            "shared_scene_bbox": True,
            "render_ratio": self.render_ratio,
            "explore_weight": self.explore_weight,
            "path_length_factor": float(self.path_length_factor),
            "geometry_weight": self.geometry_weight if self.method.startswith("defect") else 0.0,
            "geometry_thresholds": asdict(self.thresholds),
            "depth_discontinuity_gate": self.method != "defect_no_gate",
        }

    @torch.no_grad()
    def cal_utility(self, gaussian_map, voxel_map, candidates, simulator):
        if candidates.ndim != 3 or tuple(candidates.shape[1:]) != (4, 4) or len(candidates) == 0:
            raise ValueError("view candidates must be a nonempty [N,4,4] tensor")
        if not bool(torch.isfinite(candidates).all()):
            raise ValueError("view candidate poses must be finite")
        self._synchronize()
        started = time.perf_counter()
        self.last_diagnostics = self._new_diagnostics()
        self.last_diagnostics["candidate_poses"] = _json_value(candidates)
        self._heatmaps = []
        count = len(candidates)

        if self.method == "random_matched":
            utilities = torch.rand(count)
            self.last_diagnostics.update({
                "exploration": None, "uncertainty": None, "defect": None,
                "base": None, "geometry_measured": False,
                "geometry_active": False, "utility": utilities.tolist(),
            })
        else:
            resolution = np.round(self.render_ratio * np.asarray(simulator.resolution)).astype(int)
            height, width = (int(value) for value in resolution)
            if height < 1 or width < 1:
                raise ValueError("render resolution must be positive")
            bounds = torch.as_tensor(simulator.depth_range, device="cpu",
                                     dtype=torch.float32).reshape(-1)
            if bounds.numel() != 2 or not bool(torch.isfinite(bounds).all()) or not bool(bounds[1] > bounds[0]):
                raise ValueError("simulator depth range must contain finite near < far")
            near, far = bounds.tolist()
            poses = candidates.to(self.device)
            intrinsics = torch.as_tensor(simulator.intrinsic).unsqueeze(0).repeat(count, 1, 1).to(self.device)
            renderer = GaussianRenderer(
                poses, intrinsics, gaussian_map.get_attr(), gaussian_map.background_color,
                (gaussian_map.scene_near, gaussian_map.scene_far),
                (height, width), self.device,
            )
            if len(voxel_map.voxel_centers) < 1:
                raise RuntimeError("voxel map contains no cells")
            exploration, uncertainty, defects = [], [], []
            valid_fractions, positive_fractions = [], []
            geometry_measured = self.method.startswith("defect")
            for index in range(count):
                _, depth, normal, opacity, d2n, confidence, *_ = renderer.render_view(index)
                depths = depth[0]
                confidences = confidence[0]
                # A missing current-map surface exposes unknown voxels up to
                # sensor range. No future mask is obtained from the simulator.
                # nan_to_num is out-of-place; its fresh result can be changed
                # without cloning the renderer's image first.
                depth_voxel = torch.nan_to_num(depths, nan=0.0,
                                               posinf=far, neginf=0.0)
                depth_voxel[depth_voxel < 0.001] = 10000.0
                depth_voxel = depth_voxel.clamp(min=near, max=far)
                visible = voxel_map.cal_visible_mask(poses[index], intrinsics[index], depth_voxel)
                exploration.append((visible & voxel_map.unexplored_mask).float().sum()
                                   / len(voxel_map.voxel_centers))

                confidence_for_base = torch.nan_to_num(confidences, nan=1.0,
                                                       posinf=1.0, neginf=1.0)
                confidence_for_base[depths > far] = 1.0
                depth_surface = torch.nan_to_num(depths, nan=0.0,
                                                 posinf=far, neginf=0.0)
                depth_surface[depth_surface < 0.001] = far * 0.5
                uncertainty.append(((1 - confidence_for_base) * depth_surface / far).mean())
                if geometry_measured:
                    result = geometry_defect(
                        depth, normal, d2n, opacity, confidence, bounds,
                        thresholds=self.thresholds,
                        use_depth_gate=self.method != "defect_no_gate",
                    )
                    defects.append(result["score"])
                    valid_fractions.append(result["valid_fraction"])
                    positive_fractions.append(result["positive_fraction"])
                    # Keep about 6.5 MB on the computation device; only the
                    # selected image is moved to CPU in timed diagnostic IO.
                    # This avoids 100 diagnostic transfers/synchronizations.
                    self._heatmaps.append(result["heatmap"].detach())
                else:
                    defects.append(torch.zeros((), device=self.device))

            exploration = nonnegative_finite(torch.stack(exploration)).cpu()
            uncertainty = nonnegative_finite(torch.stack(uncertainty)).cpu()
            defects = nonnegative_finite(torch.stack(defects)).cpu()
            base = self.explore_weight * exploration + uncertainty
            weight = self.geometry_weight if geometry_measured else 0.0
            utilities = combine_utilities(base, defects, weight=weight,
                                           minimum_peak=self.thresholds.minimum_peak)
            active = bool(weight > 0 and defects.max() >= self.thresholds.minimum_peak
                          and defects.sum() > 0)
            self.last_diagnostics.update(_json_value({
                "render_resolution": [height, width],
                "exploration": exploration, "uncertainty": uncertainty,
                "defect": defects, "base": base,
                "base_normalized": normalize_sum(base),
                "defect_normalized": normalize_sum(defects) if active else torch.zeros_like(defects),
                "utility": utilities, "geometry_measured": geometry_measured,
                "geometry_active": active,
                "defect_max": defects.max(),
                "valid_fraction": torch.stack(valid_fractions) if valid_fractions else None,
                "positive_fraction": torch.stack(positive_fractions) if positive_fractions else None,
            }))
        self._synchronize()
        elapsed = time.perf_counter() - started
        self.last_diagnostics["utility_seconds"] = elapsed
        return utilities, elapsed

    def cal_view_scores(self, view_utilities, path_lengths):
        lengths = torch.as_tensor(path_lengths, dtype=view_utilities.dtype,
                                  device=view_utilities.device)
        reachable = torch.isfinite(lengths) & (lengths >= 0)
        # Recompute the two normalizations on reachable views, so an
        # unreachable high-defect candidate cannot absorb geometry weight.
        if self.last_diagnostics.get("base") is not None:
            self.last_diagnostics["utility_before_reachability"] = _json_value(view_utilities)
            base = torch.as_tensor(self.last_diagnostics["base"], dtype=view_utilities.dtype,
                                   device=view_utilities.device)
            defect = torch.as_tensor(self.last_diagnostics["defect"], dtype=view_utilities.dtype,
                                     device=view_utilities.device)
            base = torch.where(reachable, base, torch.zeros_like(base))
            defect = torch.where(reachable, defect, torch.zeros_like(defect))
            weight = self.geometry_weight if self.method.startswith("defect") else 0.0
            view_utilities = combine_utilities(base, defect, weight, self.thresholds.minimum_peak)
            active = bool(weight > 0 and defect.max() >= self.thresholds.minimum_peak
                          and defect.sum() > 0)
            self.last_diagnostics.update(_json_value({
                "utility": view_utilities, "normalization_domain": "reachable_candidates",
                "base_normalized": normalize_sum(base),
                "defect_normalized": normalize_sum(defect) if active else torch.zeros_like(defect),
                "geometry_active": active,
            }))
        scores = view_scores(view_utilities, lengths, self.path_length_factor)
        costs = torch.zeros_like(lengths)
        costs[reachable] = lengths[reachable] / lengths[reachable].sum().clamp_min(torch.finfo(lengths.dtype).tiny)
        self.last_diagnostics.update(_json_value({
            "path_lengths": lengths, "path_cost_normalized": costs,
            "reachable": reachable, "final_scores": scores,
            "selected_index": int(torch.argmax(scores)),
            "all_zero_utility_fallback": bool(nonnegative_finite(view_utilities)[reachable].sum() == 0),
        }))
        return scores

    def _write_diagnostics(self):
        started = time.perf_counter()
        try:
            if self.diagnostics_dir is not None:
                self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
                stem = f"step_{int(self.last_diagnostics['step']):04d}"
                selected = self.last_diagnostics.get("selected_index")
                if selected is not None and len(self._heatmaps) > selected:
                    heat = self._heatmaps[selected].cpu().numpy()
                    # Fixed [0,1] color scale preserves magnitude across steps.
                    rgb = np.stack((heat, np.sqrt(heat) * 0.7,
                                    np.maximum(0, 0.4 - heat) * (heat > 0)), axis=-1)
                    pixels = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                    image_path = self.diagnostics_dir / f"{stem}_selected_heatmap.png"
                    Image.fromarray(pixels).save(image_path)
                    self.last_diagnostics["selected_heatmap"] = image_path.name
                    self.last_diagnostics["heatmap_scale"] = [0.0, 1.0]
                    self.last_diagnostics["selected_heatmap_max"] = float(heat.max())
                # The persisted field excludes its own JSON write. The full
                # duration including JSON and atomic replace is recorded in
                # the cumulative runner counter and last_diagnostics below.
                self.last_diagnostics["diagnostics_seconds"] = time.perf_counter() - started
                path = self.diagnostics_dir / f"{stem}.json"
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(self.last_diagnostics, ensure_ascii=False,
                                                 indent=2, allow_nan=False), encoding="utf-8")
                temporary.replace(path)
        finally:
            elapsed = time.perf_counter() - started
            self.diagnostics_io_seconds += elapsed
            self.last_diagnostics["diagnostics_seconds"] = elapsed
            self.last_diagnostics["diagnostics_io_seconds_cumulative"] = self.diagnostics_io_seconds

    def generate_random_candidates(self, voxel_map, num):
        if num <= 0:
            return torch.empty((0, 4, 4), dtype=torch.float32)
        centers = voxel_map.voxel_centers.cpu().numpy()
        available = voxel_map.free_mask_w_margin.cpu().numpy()
        available &= np.linalg.norm(centers - self.pose[:3, 3].numpy(), axis=1) <= self.radius
        if not np.any(available):
            raise RuntimeError("no free candidate positions within planning radius")
        return super().generate_random_candidates(voxel_map, num)

    def plan(self, map, simulator, recorder):
        """Retain the upstream recipe while replacing its debugger failure path."""
        gaussian_map, voxel_map = map
        self._plan_calls += 1
        self.last_diagnostics = self._new_diagnostics()
        self._heatmaps = []
        planning_time = 0.0
        try:
            if self.init:
                self._synchronize()
                started = time.perf_counter()
                voxel_map.update_graph(self.get_robot_space(voxel_map))
                roi = torch.empty((0, 4, 4), dtype=torch.float32)
                if self.max_roi_sample_num > 0:
                    voxel_map.update_utility(gaussian_map, self.use_confidence)
                    generated = self.generate_roi_candidates(voxel_map, self.max_roi_sample_num)
                    if len(generated):
                        roi = generated[:self.max_roi_sample_num]
                remaining = self.sample_num - len(roi)
                random = (self.generate_random_candidates(voxel_map, remaining) if remaining > 0
                          else torch.empty((0, 4, 4), dtype=torch.float32))
                candidates = torch.cat((roi, random), dim=0)
                if len(candidates) == 0:
                    raise RuntimeError("candidate generation produced no views")
                self._synchronize()
                sampling_seconds = time.perf_counter() - started
                planning_time += sampling_seconds
                self.q_planner2gui.put(Planner2Gui(candidates[:, :3, 3], candidates[:, :3, 2]))
                utilities, utility_seconds = self.cal_utility(gaussian_map, voxel_map, candidates, simulator)
                planning_time += utility_seconds
                self.last_diagnostics.update({"roi_candidates": len(roi), "random_candidates": len(random),
                                              "sampling_seconds": sampling_seconds})
                started = time.perf_counter()
                paths, lengths = self.path_planner.search_goal(self.pose[:3, 3].numpy(),
                                                              candidates[:, :3, 3].numpy(), voxel_map)
                scores = self.cal_view_scores(utilities, lengths)
                selected = int(torch.argmax(scores))
                nbv = candidates[selected]
                waypoints = voxel_map.index_2_xyz(paths[selected]).cpu()
                if len(waypoints) == 0:
                    # The already occupied start/goal voxel has zero travel.
                    if float(lengths[selected]) != 0:
                        raise RuntimeError("reachable candidate has no path waypoints")
                    waypoints = torch.stack((self.pose[:3, 3], nbv[:3, 3]))
                path_seconds = time.perf_counter() - started
                planning_time += path_seconds
                self.last_diagnostics["path_scoring_seconds"] = path_seconds
            else:
                nbv = torch.eye(4)
                nbv[:3, :3] = self.pose[:3, :3]
                index = voxel_map.xyz_2_index(self.pose[:3, 3])
                nbv[:3, 3] = voxel_map.index_2_xyz([index])[0].cpu()
                waypoints = torch.stack((self.pose[:3, 3], nbv[:3, 3]))
                self.init = True
                self.last_diagnostics.update({"initialization": True,
                                              "candidate_poses": [], "selected_index": None})

            if torch.allclose(self.pose, nbv):
                # Upstream wp2path samples zero poses for a zero-motion action.
                # Returning the chosen pose still yields a legitimate sensor
                # observation and prevents path[-1] from indexing an empty list.
                path, path_length = nbv.unsqueeze(0), 0.0
            else:
                path, path_length = wp2path(self.pose[:3, :3], nbv[:3, :3], waypoints)
                if len(path) == 0:
                    path = nbv.unsqueeze(0)
            # Upstream linspace(0, 1, 1) returns only the initial pose for a
            # small rotation/displacement. The actual sensor pose must still
            # reach the selected candidate, including its full orientation.
            endpoint = nbv.to(path)
            if not torch.allclose(path[-1], endpoint, atol=1e-6, rtol=0):
                path = torch.cat((path, endpoint.unsqueeze(0)), dim=0)
            self.pose = nbv
            self.last_diagnostics.update(_json_value({"status": "selected", "selected_pose": nbv,
                                                      "travel_distance": path_length,
                                                      "planning_seconds": planning_time}))
            self._write_diagnostics()
            if recorder is not None:
                recorder.update_time("planning", planning_time)
                recorder.update_time("flight", cal_flight_time(path_length, flight_speed=self.flight_speed))
                recorder.update_path(path, path_length)
            return path
        except Exception as error:
            self.last_diagnostics.update({"status": "failed", "error_type": type(error).__name__,
                                          "error": str(error)})
            self._write_diagnostics()
            raise
