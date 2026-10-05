"""Export trusted local ActiveGS artifacts for the browser; no GPU required.

Only read pickle files produced by the project's own ActiveGS runs. The preview
is decimated; evaluation numbers always refer to the original full-size meshes.
"""
import argparse
import io
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d


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


def export(run, output, faces=80000):
    experiment = run / "experiments/original/replica/office0/confidence/0"
    metrics = json.loads((experiment / "final_result.json").read_text())
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
        all_cameras = cameras
        checkpoints.append({"step": step, "time": metrics["time"][i],
                            "path_length": metrics["path_length"][i], "mesh": name,
                            "original": original,
                            "preview": {"vertices": len(mesh.vertices), "faces": len(mesh.triangles)},
                            "camera_count": len(cameras),
                            "accuracy_cm": metrics["mesh_accuracy"][i],
                            "completion_cm": metrics["mesh_completion"][i],
                            "coverage_percent": metrics["mesh_completion_ratio"][i],
                            "chamfer_mm": metrics["mesh_chamfer_distance"][i] * 1000})
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
    if positions and len(stops) != len(all_cameras):
        raise ValueError("Motion-path endpoints do not match acquisition cameras")
    if not positions:
        positions = [item["position"] for item in all_cameras]
        stops = list(range(1, len(positions) + 1))
    rgb = run / "evidence/initial_rgb.png"
    if rgb.exists():
        shutil.copyfile(rgb, output / "initial_rgb.png")
    manifest = {"id": run.name, "scene": "Replica office0", "method": "ActiveGS · confidence",
                "status": "completed", "bounds": bounds, "up_axis": "z",
                "camera_convention": "OpenCV camera-to-world",
                "trajectory_kind": "motion_path" if path_file.exists() else "keyframe_connections",
                "trajectory": positions, "path_stops": stops, "cameras": all_cameras,
                "checkpoints": checkpoints, "initial_rgb": "initial_rgb.png" if rgb.exists() else None,
                "scope": "原始 baseline 单次执行验证；自研视角评分尚未实现。",
                "preview_note": "浏览器网格已简化；指标由原始完整网格计算。"}
    encoded = json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    temp = output / "manifest.json.tmp"
    temp.write_text(encoded, encoding="utf-8")
    temp.replace(output / "manifest.json")
    print(json.dumps({"run": run.name, "checkpoints": len(checkpoints), "cameras": len(all_cameras),
                      "path_positions": len(positions), "bounds": bounds}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--faces", type=int, default=80000)
    args = parser.parse_args()
    if args.faces < 1000:
        parser.error("--faces must be >= 1000")
    export(args.run.resolve(), args.output.resolve(), args.faces)
