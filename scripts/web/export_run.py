"""Export trusted local ActiveGS artifacts for the browser; no GPU required.

Only read pickle files produced by the project's own ActiveGS runs. The preview
is decimated; evaluation numbers always refer to the original full-size meshes.
"""
import argparse
import hashlib
import io
import json
import math
import pickle
import shutil
from pathlib import Path

import numpy as np

METHOD_LABELS = {
    "confidence": "ActiveGS · confidence（原作者流程）",
    "confidence_nooracle": "Confidence · 关闭候选真值掩码",
    "random_matched": "Random · 匹配候选与路径成本",
    "defect": "ViewMend3D · 几何缺陷评分",
    "defect_no_gate": "Defect · 移除深度跳变门控",
    "refine_only": "Refine only · 仅优化已有观测",
}


def resolve_experiment(run, experiment=None):
    """Keep the original default; a multi-branch batch needs an explicit path."""
    if experiment is not None:
        target = experiment if experiment.is_absolute() else run / experiment
        target = target.resolve()
        target.relative_to(run.resolve())
    else:
        target = run / "experiments/original/replica/office0/confidence/0"
        if not (target / "final_result.json").is_file():
            matches = list((run / "experiments/benchmark").glob("replica/*/*/*/final_result.json"))
            if len(matches) != 1:
                raise ValueError("Specify --experiment for a batch with multiple or no completed branches")
            target = matches[0].parent
    if not (target / "final_result.json").is_file():
        raise ValueError(f"No completed metrics in {target}")
    return target


def experiment_metadata(run, experiment):
    protocol_file = experiment / "protocol.json"
    relative = experiment.relative_to(run).as_posix()
    if protocol_file.exists():
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        if not isinstance(protocol, dict):
            raise ValueError("protocol.json must be an object")
        method = protocol.get("method", protocol.get("method_id"))
        seed = protocol.get("seed")
        if method not in METHOD_LABELS or type(seed) is not int:
            raise ValueError("protocol.json must identify the method and integer seed")
        scene = protocol.get("scene", "office0")
        scope = protocol.get("scope", "共享采集前缀与预算的对照实验；质量提升须由完成的结果判断。")
    else:
        if "/benchmark/" in "/" + relative:
            raise ValueError("Benchmark branches require protocol.json")
        method, seed, scene = "confidence", None, "office0"
        protocol = {"name": "original", "candidate_oracle_mask": True,
                    "rng_seed_fixed": False, "budget_type": "mission_time"}
        scope = "原作者流程的单次执行验证，含候选真值掩码；与公平对照分开展示。"
    comparison = {"protocol": protocol.get("protocol", protocol),
                  "scene": scene,
                  "scene_mesh_sha256": protocol.get("scene_mesh_sha256"),
                  "scene_assets_sha256": protocol.get("scene_assets_sha256"),
                  "context_hash": protocol.get("context_hash"),
                  "prefix_sha256": protocol.get("prefix_sha256"),
                  "prefix_camera_sha256": protocol.get("prefix_camera_sha256"),
                  "eval_seed": protocol.get("eval_seed"),
                  "source_versions": protocol.get("source_versions")}
    comparison_id = hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest()[:12]
    return {"schema_version": 2, "run_id": run.name, "experiment": relative,
            "scene": f"Replica {str(scene).split('/')[-1]}", "method_id": method,
            "method": METHOD_LABELS[method], "seed": seed, "protocol": protocol,
            "scope": scope, "comparison_id": protocol.get("comparison_id", comparison_id)}


def count_for_checkpoint(metrics, i, camera_count):
    observation_count = metrics.get("observation_count", [])[i] if "observation_count" in metrics else camera_count
    update_event = metrics.get("update_event", metrics["step"])[i]
    # The author's evaluator serializes np.loadtxt IDs as 40.0 rather than 40.
    # Accept exactly integral legacy IDs while controlled arrays remain strict.
    if "update_event" not in metrics and isinstance(update_event, float) and math.isfinite(update_event) and update_event.is_integer():
        update_event = int(update_event)
    if type(observation_count) is not int or observation_count != camera_count:
        raise ValueError("Recorded observation count must match acquisition cameras")
    if type(update_event) is not int or update_event < observation_count:
        raise ValueError("Update event must be an integer >= observation count")
    return observation_count, update_event


def validate_motion_path(positions, stops, cameras):
    if positions and len(stops) != len(cameras):
        raise ValueError("Motion-path endpoints do not match acquisition cameras")
    for stop, camera in zip(stops, cameras):
        if not 1 <= stop <= len(positions) or not np.allclose(positions[stop - 1], camera["position"], atol=1e-4, rtol=0):
            raise ValueError("Motion-path endpoints do not align with acquisition camera positions")


def export_diagnostics(experiment, output, event, run=None):
    """Export only actual planner diagnostics; absence never creates a heatmap."""
    folder = experiment / "diagnostics"
    matches = [folder / f"step_{event:03}.json", folder / f"event_{event:03}.json",
               folder / f"{event:03}.json", folder / f"step_{event:04}.json"]
    source = next((path for path in matches if path.is_file()), None)
    if source is None and run is not None:
        protocol = json.loads((experiment / "protocol.json").read_text(encoding="utf-8")) if (experiment / "protocol.json").exists() else {}
        prefix = protocol.get("protocol", {}).get("prefix", 0)
        if type(prefix) is int and event <= prefix and type(protocol.get("seed")) is int:
            folder = run / "common-prefix" / f"seed_{protocol['seed']}" / "diagnostics"
            source = folder / f"step_{event:04}.json"
            if not source.is_file():
                source = None
    if source is None:
        return None
    diagnostic = json.loads(source.read_text(encoding="utf-8"))
    selected = diagnostic.get("selected_index")
    candidates = diagnostic.get("candidates", [])
    poses = diagnostic.get("candidate_poses", [])
    count = len(candidates) if candidates else len(poses)
    if type(selected) is not int or not 0 <= selected < count:
        return None
    if candidates:
        candidate = candidates[selected]
    else:
        candidate = {"pose": poses[selected]}
        fields = {"exploration": "E", "uncertainty": "U", "defect": "D", "base": "B",
                  "utility": "utility", "path_lengths": "path_length", "final_scores": "final_score",
                  "valid_fraction": "valid_fraction", "positive_fraction": "positive_fraction"}
        for source_key, target_key in fields.items():
            values = diagnostic.get(source_key)
            if isinstance(values, list) and selected < len(values):
                candidate[target_key] = values[selected]
    # Browser needs the selected candidate, not a potentially huge candidate set.
    result = {"event": event, "selected_index": selected,
              "geometry_active": diagnostic.get("geometry_active"),
              "geometry_measured": diagnostic.get("geometry_measured"),
              "candidate_count": count, "candidate": candidate,
              "heatmap_scale": diagnostic.get("heatmap_scale"),
              "heatmap_max": diagnostic.get("selected_heatmap_max")}
    heatmap_ref = diagnostic.get("selected_heatmap", diagnostic.get("heatmap", diagnostic.get("heatmap_path")))
    possible = [folder / f"step_{event:03}_heatmap.png", folder / f"event_{event:03}_heatmap.png",
                folder / f"heatmap_{event:03}.png"]
    if isinstance(heatmap_ref, str):
        possible.insert(0, source.parent / heatmap_ref)
    heatmap = next((path for path in possible if path.is_file()), None)
    if heatmap is not None:
        heatmap.resolve().relative_to(source.parent.resolve())
        name = f"diagnostic_{event:03}.png"
        shutil.copyfile(heatmap, output / name)
        result["heatmap"] = name
    return result


def load_local_path(path):
    # PyTorch's pickle storage hook otherwise restores tensors to their old GPU.
    import torch

    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda blob: torch.load(io.BytesIO(blob), map_location="cpu")
            return super().find_class(module, name)

    with path.open("rb") as stream:
        return CPUUnpickler(stream).load()


def export(run, output, faces=80000, experiment=None):
    import open3d as o3d

    experiment = resolve_experiment(run, experiment)
    metadata = experiment_metadata(run, experiment)
    metrics = json.loads((experiment / "final_result.json").read_text(encoding="utf-8"))
    required = ("step", "time", "path_length", "mesh_accuracy", "mesh_completion",
                "mesh_completion_ratio", "mesh_chamfer_distance")
    count = len(metrics.get("step", []))
    if not count or any(len(metrics.get(key, [])) != count for key in required):
        raise ValueError("Completed metrics must have nonempty, aligned checkpoint arrays")
    if any(key in metrics and len(metrics[key]) != count for key in ("observation_count", "update_event")):
        raise ValueError("Observation and update arrays must align with checkpoint metrics")
    output.mkdir(parents=True, exist_ok=True)
    all_cameras = []
    checkpoints = []
    bounds = None
    for i, step in enumerate(metrics["step"]):
        step = int(step)
        source = experiment / "map" / f"mesh_{step:03}.ply"
        mesh = o3d.io.read_triangle_mesh(str(source))
        original = {"vertices": len(mesh.vertices), "faces": len(mesh.triangles)}
        if original["faces"] == 0:
            raise ValueError(f"Empty mesh: {source}")
        bounds = [mesh.get_min_bound().tolist(), mesh.get_max_bound().tolist()]
        if original["faces"] > faces:
            mesh = mesh.simplify_quadric_decimation(faces)
        name = f"mesh_{step:03}.ply"
        if not o3d.io.write_triangle_mesh(str(output / name), mesh):
            raise ValueError(f"Failed to write {name}")
        with (experiment / "map" / f"cameras_{step:03}.pkl").open("rb") as stream:
            params = pickle.load(stream)
        cameras = []
        for row in params:
            pose = np.asarray(row[:16]).reshape(4, 4)
            cameras.append({"position": pose[:3, 3].tolist(),
                            "rotation": pose[:3, :3].reshape(-1).tolist(),
                            "intrinsic": row[16:]})
        if len(cameras) < len(all_cameras):
            raise ValueError("Acquisition camera count cannot decrease across checkpoints")
        all_cameras = cameras
        observation_count, update_event = count_for_checkpoint(metrics, i, len(cameras))
        checkpoints.append({"step": step, "time": metrics["time"][i],
                            "path_length": metrics["path_length"][i], "mesh": name,
                            "original": original,
                            "preview": {"vertices": len(mesh.vertices), "faces": len(mesh.triangles)},
                            "camera_count": len(cameras), "observation_count": observation_count,
                            "update_event": update_event,
                            "accuracy_cm": metrics["mesh_accuracy"][i],
                            "completion_cm": metrics["mesh_completion"][i],
                            "coverage_percent": metrics["mesh_completion_ratio"][i],
                            "chamfer_mm": metrics["mesh_chamfer_distance"][i] * 1000,
                            "diagnostic": export_diagnostics(experiment, output, update_event, run)})
    positions = []
    stops = []
    path_file = experiment / "global_path.pkl"
    if path_file.exists():
        path = load_local_path(path_file)
        for item in path.values():
            pose = np.asarray(item["pose"]).reshape(4, 4)
            positions.append(pose[:3, 3].tolist())
            if item["name"] is not None:
                stops.append(len(positions))
    # Prefix endpoints map directly to acquisition keyframes, never test poses.
    validate_motion_path(positions, stops, all_cameras)
    prefix = metadata["protocol"].get("protocol", {}).get("prefix")
    if metadata["method_id"] == "refine_only" and type(prefix) is int and len(all_cameras) != prefix:
        raise ValueError("Refine-only branch must preserve the prefix acquisition count")
    if not positions:
        positions = [item["position"] for item in all_cameras]
        stops = list(range(1, len(positions) + 1))
    rgb = experiment / "initial_rgb.png"
    if not rgb.exists():
        rgb = run / "evidence/initial_rgb.png"
    if rgb.exists():
        shutil.copyfile(rgb, output / "initial_rgb.png")
    manifest = {**metadata, "id": output.name,
                "status": "completed", "bounds": bounds, "up_axis": "z",
                "camera_convention": "OpenCV camera-to-world",
                "trajectory_kind": "motion_path" if path_file.exists() else "keyframe_connections",
                "trajectory": positions, "path_stops": stops, "cameras": all_cameras,
                "checkpoints": checkpoints, "initial_rgb": "initial_rgb.png" if rgb.exists() else None,
                "cost": metrics.get("cost", metadata["protocol"].get("cost", {})),
                "evaluation": metrics.get("evaluation"),
                "plot_axis": "update_event" if "update_event" in metrics and metadata["protocol"].get("protocol", {}).get("mode") != "time" else "time",
                "preview_note": "浏览器网格已简化；指标由原始完整网格计算。"}
    encoded = json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    temp = output / "manifest.json.tmp"
    temp.write_text(encoded, encoding="utf-8")
    temp.replace(output / "manifest.json")
    summary = {key: manifest[key] for key in ("id", "run_id", "scene", "method", "method_id", "seed",
               "protocol", "scope", "comparison_id", "status", "cost", "evaluation")}
    summary["final"] = {key: value for key, value in checkpoints[-1].items() if key != "diagnostic"}
    summary_temp = output / "summary.json.tmp"
    summary_temp.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    summary_temp.replace(output / "summary.json")
    print(json.dumps({"run": run.name, "checkpoints": len(checkpoints), "cameras": len(all_cameras),
                      "path_positions": len(positions), "bounds": bounds}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--faces", type=int, default=80000)
    parser.add_argument("--experiment", type=Path, help="Completed experiment directory, absolute or relative to run")
    args = parser.parse_args()
    if args.faces < 1000:
        parser.error("--faces must be >= 1000")
    export(args.run.resolve(), args.output.resolve(), args.faces, args.experiment)
