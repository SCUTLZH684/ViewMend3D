"""Exercise the original simulator at its configured initial camera pose."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source = args.source.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(source))
os.chdir(source)

import hydra
import numpy as np
from PIL import Image
import torch
from simulator import get_simulator

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
with hydra.initialize_config_dir(config_dir=str(source / "config"), version_base=None):
    config = hydra.compose(config_name="main", overrides=["planner=confidence", "scene=replica/office0", "use_gui=false"])
simulator = get_simulator(config)
frame = simulator.simulate(torch.tensor([list(row) for row in config.planner.init_pose], dtype=torch.float32), require_gt=True)
rgb = frame["rgb"].permute(1, 2, 0).numpy()
depth = frame["depth"].squeeze().numpy()
valid = np.isfinite(depth) & (depth > 0)
if rgb.shape != (512, 512, 3) or depth.shape != (512, 512) or not valid.any() or rgb.std() <= 0:
    raise RuntimeError("Simulator returned invalid/empty observations")
Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(output / "initial_rgb.png")
np.save(output / "initial_depth.npy", depth)
preview = np.uint8(np.where(valid, np.clip(depth / 5, 0, 1) * 255, 0))
Image.fromarray(preview).save(output / "initial_depth_preview.png")
report = {"simulator_render_passed": True, "torch_version": torch.__version__,
          "torch_cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
          "scene": config.scene.scene_name, "rgb_shape": list(rgb.shape),
          "depth_shape": list(depth.shape), "valid_depth_fraction": float(valid.mean()),
          "positive_depth_range_m": [float(depth[valid].min()), float(depth[valid].max())],
          "has_missing_surface": bool(config.scene.has_missing_surface)}
(output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
simulator.sim.close()
