"""Check actual upstream state restoration on CPU, without importing CUDA side effects.

The pinned mapping.utils initializes LPIPS on CUDA at import. Extract the actual
class/function definitions instead; their method bodies remain unchanged. This
checks serialization and interfaces, not rendering, mapping or mesh quality.
Run in the configured ActiveGS environment as a standalone process.
"""
import argparse
import ast
from collections import defaultdict
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import types


def extract(path, names, module_name, dependencies):
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    module.__dict__.update(dependencies)
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError(f"Required upstream definitions missing in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from scipy.ndimage import binary_dilation, generate_binary_structure
    from hydra import compose, initialize_config_dir
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from viewmend3d.benchmark import object_state, load_prefix, gradient_refinement_only
    assert not torch.cuda.is_initialized()
    for name in ("mapping", "utils"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    operations = extract(upstream / "utils/operations.py", {"inverse_sigmoid"},
                         "viewmend_audit_ops", {"torch": torch})
    gaussian_module = extract(upstream / "mapping/gaussian_map.py", {"GaussianMap"},
                              "mapping.gaussian_map", {"torch": torch, "nn": torch.nn,
                              "inverse_sigmoid": operations.inverse_sigmoid})
    voxel_module = extract(upstream / "mapping/voxel_map.py", {"VoxelMap", "VoxelGrpah"},
                           "mapping.voxel_map", {"torch": torch, "np": np,
                           "defaultdict": defaultdict, "binary_dilation": binary_dilation,
                           "generate_binary_structure": generate_binary_structure})
    recorder_module = extract(upstream / "utils/common.py", {"MissionRecorder"}, "utils.common",
                              {"torch": torch, "np": np, "os": os, "pickle": pickle})
    with initialize_config_dir(config_dir=str(upstream / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=["planner=confidence", "scene=replica/office0",
                                                     "use_gui=false", "debug=false"])
    device = torch.device("cpu")
    gaussian = gaussian_module.GaussianMap(cfg.mapper.gaussian_map, device)
    shapes = {"_means": (3, 3), "_scales": (3, 3), "_rotations": (3, 4), "_opacities": (3,),
              "_harmonics": (3, 1, 3), "view_scores": (3,), "view_supports": (3,), "view_means": (3, 3)}
    for name, shape in shapes.items():
        setattr(gaussian, name, torch.arange(np.prod(shape)).reshape(shape).float())
    gaussian.training_data = [{"rgb": torch.arange(12).reshape(3, 2, 2).float(),
                               "depth": torch.tensor([[[1., -1.], [-2., 3.]]]),
                               "extrinsic": torch.eye(4), "intrinsic": torch.eye(3),
                               "depth_range": torch.tensor([0., 5.])}]
    gaussian.training_performance = torch.tensor([.25])
    gaussian.is_init = True
    gaussian.init_training()
    bbox = np.array([[0., 0., 0.], [.6, .6, .6]])
    voxel = voxel_module.VoxelMap(cfg.mapper.voxel_map, bbox, device)
    voxel.voxel_lo.fill_(-4)
    voxel.unexplored_mask[0] = False
    voxel.update_graph(torch.zeros(len(voxel.voxel_centers), dtype=torch.bool))
    with tempfile.TemporaryDirectory(prefix="viewmend-prefix-cpu-") as temporary:
        recorder = recorder_module.MissionRecorder(temporary, cfg.experiment)
        recorder.time_dict = {"mapping": 1.2, "planning": 2.3, "flight": 3.4}
        recorder.camera_params_list = [[float(i) for i in range(25)]]
        recorder.update_path(torch.eye(4).unsqueeze(0), .7)
        state = {"gaussian": object_state(gaussian, ("optimizer",)), "voxel": object_state(voxel),
                 "bbox": bbox, "recorder": object_state(recorder, ("save_dir",)),
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch_cpu": torch.random.get_rng_state(), "torch_cuda": []}}
        path = Path(temporary) / "prefix.th"
        torch.save(state, path)
        _, g1, v1, r1 = load_prefix(path, cfg, device, Path(temporary) / "branch-1")
        _, g2, _, _ = load_prefix(path, cfg, device, Path(temporary) / "branch-2")
        for name in (*shapes, "training_performance"):
            assert torch.equal(getattr(g1, name), getattr(gaussian, name)), name
            assert getattr(g1, name).data_ptr() != getattr(g2, name).data_ptr(), name
        for name, value in gaussian.training_data[0].items():
            assert torch.equal(g1.training_data[0][name], value), name
            assert g1.training_data[0][name].data_ptr() != g2.training_data[0][name].data_ptr(), name
        for name in ("voxel_lo", "unexplored_mask", "voxel_centers", "dim", "size", "bbox"):
            assert torch.equal(getattr(v1, name), getattr(voxel, name)), name
        assert np.array_equal(v1.graph.previous_traversable_mask, voxel.graph.previous_traversable_mask)
        assert dict(v1.graph.dense_graph) == dict(voxel.graph.dense_graph)
        assert r1.camera_params_list == recorder.camera_params_list
        assert r1.time_dict == recorder.time_dict and r1.pose_id == recorder.pose_id
        assert torch.equal(r1.global_path_dict[0]["pose"], recorder.global_path_dict[0]["pose"])
        assert r1.save_dir.endswith("branch-1") and not hasattr(g1, "optimizer")
        g1.init_training()
        assert all(isinstance(p, torch.nn.Parameter) and p.requires_grad for p in g1.get_params())
        assert len(g1.optimizer.param_groups) == 5 and not g1.optimizer.state
        g1.get_means.sum().backward()
        g1.optimizer.step()
        assert not torch.equal(g1.get_means, g2.get_means)
        original = g1.post_processing
        with gradient_refinement_only(g1):
            g1.post_processing()
        assert g1.post_processing == original
        assert not torch.cuda.is_initialized()
    print("PASS: actual-class CPU prefix round-trip; Gaussian tensors, RGB-D/cameras, voxel graph, "
          "recorder, isolated branches, Adam and post-processing restoration; CUDA not initialized")


if __name__ == "__main__":
    main()
