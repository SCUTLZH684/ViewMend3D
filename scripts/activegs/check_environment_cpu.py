"""Inspect pinned sources, package metadata and scene files without importing GPU modules.

This is an input/environment check, not a CUDA, EGL or reconstruction test.
Run with the configured ActiveGS Python before waiting for an idle GPU.
"""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from viewmend3d.input_assets import scene_assets

ACTIVEGS_COMMIT = "558121a00b2eca84d9851a609e99ebb8e26ea2d8"
PINS = {"torch": "2.1.2", "torchvision": "0.16.2", "numpy": "1.26.4", "open3d": "0.17.0",
        "trimesh": "4.4.1", "lpips": "0.1.4", "habitat-sim": "0.2.4"}


def source_check(upstream):
    def git(*args):
        return subprocess.run(["git", *args], cwd=upstream, text=True, capture_output=True,
                              check=True, timeout=20).stdout.strip()
    commit = git("rev-parse", "HEAD")
    if commit != ACTIVEGS_COMMIT:
        raise ValueError(f"ActiveGS commit differs from the verified baseline: {commit}")
    changed = set(git("diff", "HEAD", "--name-only").splitlines())
    if not changed <= {"main.py", "data_generation.py"}:
        raise ValueError(f"Unexpected upstream changes: {sorted(changed)}")
    for filename in changed:
        original = subprocess.run(["git", "show", f"HEAD:{filename}"], cwd=upstream, text=True,
                                  capture_output=True, check=True, timeout=20).stdout
        expected = original.replace('mp.set_start_method("spawn")', 'mp.set_start_method("spawn", force=True)')
        if (upstream / filename).read_text(encoding="utf-8") != expected:
            raise ValueError(f"Upstream change is not the recorded startup fix: {filename}")
    return {"activegs_commit": commit, "startup_fix_files": sorted(changed)}


def check(upstream, root):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    sources = source_check(upstream)
    packages = {}
    for name, expected in PINS.items():
        try:
            actual = version(name)
        except PackageNotFoundError as error:
            raise ValueError(f"Missing package metadata: {name}") from error
        if actual.split("+")[0] != expected:
            raise ValueError(f"Unverified package version: {name}={actual}; expected {expected}")
        packages[name] = actual
    with initialize_config_dir(config_dir=str(upstream / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=["planner=confidence", "scene=replica/office0",
                                                     "use_gui=false", "debug=false"])
    if (cfg.planner.sample_num != 100 or cfg.planner.max_roi_sample_num != 30
            or cfg.planner.path_length_factor != .5 or cfg.planner.render_ratio != .25
            or cfg.planner.explore_weight != 1000 or cfg.mapper.gaussian_map.optimization_steps != 10):
        raise ValueError("The source configuration differs from the frozen v1 recipe")
    assets, asset_digest = scene_assets(upstream, cfg.scene.mesh_path, cfg.scene.scene_id)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 20 * 1024 ** 3:
        raise ValueError("Less than 20 GiB disk space remains for experiment outputs")
    if "torch" in sys.modules or "habitat_sim" in sys.modules:
        raise RuntimeError("The CPU checker unexpectedly imported a GPU runtime module")
    return {"static_environment_passed": True, "scope": "CPU input checks only; CUDA/EGL/rendering unverified",
            "gpu_modules_imported": False, "sources": sources, "packages": packages,
            "scene": cfg.scene.scene_name, "scene_assets_sha256": asset_digest, "assets": assets,
            "free_disk_bytes": free_bytes, "simulator_config": OmegaConf.to_container(cfg.simulator, resolve=True),
            "source_future_candidate_mask": cfg.scene.has_missing_surface,
            "controlled_future_candidate_mask": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = check(args.upstream.resolve(), args.root.resolve())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report = {"static_environment_passed": False, "error": str(error), "scope": "CPU input checks only"}
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["static_environment_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
