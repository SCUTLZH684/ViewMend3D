"""Controlled ActiveGS experiments, separate from the author's entry point.

Imports of the CUDA implementation are intentionally delayed until GPU admission
has succeeded. All checkpoints, caches, and pickle inputs are our local outputs.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import time
from contextlib import contextmanager

from .protocol import Protocol, domain_seed
from .randomness import random_domain


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_versions(upstream):
    from importlib.metadata import version
    source = Path(upstream)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                            capture_output=True, text=True, check=True).stdout.strip()
    def tree_hash(directory):
        digest = hashlib.sha256()
        for filename in sorted(Path(directory).rglob("*.py")):
            digest.update(str(filename.relative_to(directory)).replace("\\", "/").encode())
            digest.update(bytes.fromhex(sha256_file(filename)))
        return digest.hexdigest()
    return {"activegs_commit": commit, "upstream_python_sha256": tree_hash(source),
            "viewmend_python_sha256": tree_hash(Path(__file__).parent),
            **{name: version(name) for name in ("torch", "numpy", "trimesh", "open3d")}}


def prefix_context(cfg, protocol, versions):
    from omegaconf import OmegaConf
    context = {"config": OmegaConf.to_container(cfg, resolve=True), "protocol": protocol.as_dict(),
               "source_versions": versions, "scene_mesh_sha256": sha256_file(cfg.scene.mesh_path)}
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
    return context, hashlib.sha256(encoded).hexdigest()


def require_idle_gpu(index):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, check=True, timeout=10)
    for row in result.stdout.splitlines():
        gpu, used, utilization = (int(part.strip()) for part in row.split(","))
        if gpu == index:
            if used >= 1024 or utilization > 5:
                raise RuntimeError(f"GPU {index} is occupied ({used} MiB, {utilization}%); no experiment started")
            return {"index": index, "used_mb": used, "utilization": utilization}
    raise ValueError(f"GPU {index} was not reported by nvidia-smi")


class ObservationOnlySimulator:
    """Give planners calibration/bounds, and forbid every candidate observation."""
    def __init__(self, simulator):
        self._simulator = simulator
        self._observation_allowed = False
        for name in ("resolution", "intrinsic", "depth_range", "bbox", "scene_name"):
            setattr(self, name, getattr(simulator, name))
        self.has_missing_surface = False  # disables the upstream future-mask branch
        self.observation_calls = 0
        self.blocked_calls = 0

    @contextmanager
    def acquire_selected_view(self):
        self._observation_allowed = True
        try:
            yield
        finally:
            self._observation_allowed = False

    def simulate(self, pose, valid_mask_only=False, require_gt=False):
        if not self._observation_allowed or valid_mask_only or require_gt:
            self.blocked_calls += 1
            raise RuntimeError("Controlled protocol forbids simulator measurements during planning")
        self.observation_calls += 1
        return self._simulator.simulate(pose)


def transfer_tree(value, device):
    """Copy tensor trees; preserve the upstream graph object and NumPy arrays."""
    import torch
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {key: transfer_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [transfer_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(transfer_tree(item, device) for item in value)
    return copy.deepcopy(value)


def object_state(instance, exclude=()):
    return {name: transfer_tree(value, "cpu") for name, value in vars(instance).items()
            if name not in set(exclude) | {"device"}}


@contextmanager
def gradient_refinement_only(gaussian_map):
    # Upstream post_processing also increments support counts for its latest
    # camera. Repeated gradients are not repeated observations in this control.
    original = gaussian_map.post_processing
    gaussian_map.post_processing = lambda: None
    try:
        yield
    finally:
        gaussian_map.post_processing = original


def synchronized_time(callback):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = callback()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def save_initial_rgb(frame, destination):
    import numpy as np
    from PIL import Image
    rgb = frame["rgb"].detach().cpu().permute(1, 2, 0).numpy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).save(destination)


def acquire_event(gaussian_map, voxel_map, planner, simulator, recorder, seed, event):
    planning_before = recorder.t_planning
    planner.diagnostics_step = event
    diagnostic_before = getattr(planner, "diagnostics_io_seconds", 0.0)
    with random_domain(seed, "planning", event):
        path, planning_seconds = synchronized_time(lambda: planner.plan(
            (gaussian_map, voxel_map), simulator, recorder))
    diagnostic_seconds = max(0.0, getattr(planner, "diagnostics_io_seconds", 0.0) - diagnostic_before)
    planning_seconds = max(0.0, planning_seconds - diagnostic_seconds)
    if len(path) == 0:
        raise RuntimeError("Planner returned an empty camera path")
    recorder.time_dict["planning"] = planning_before + planning_seconds
    sensor_start = time.perf_counter()
    with random_domain(seed, "sensor", event), simulator.acquire_selected_view():
        frame = simulator.simulate(path[-1].cpu())
    import torch
    if not torch.allclose(frame["extrinsic"].cpu(), planner.pose.cpu(), atol=1e-5, rtol=0):
        raise RuntimeError("Actual observation pose differs from the selected view")
    sensor_seconds = time.perf_counter() - sensor_start
    frame = {name: value.to(gaussian_map.device) for name, value in frame.items()}

    def mapping():
        gaussian_map.update(frame)
        voxel_map.update(frame)

    with random_domain(seed, "mapping", event):
        _, mapping_seconds = synchronized_time(mapping)
    recorder.update_time("mapping", mapping_seconds)
    recorder.save_dataframe(frame, f"{event:03}")
    return frame, {"event": event, "observations": len(gaussian_map.training_data),
                   "planning_seconds": planning_seconds, "mapping_seconds": mapping_seconds,
                   "sensor_seconds": sensor_seconds, "mission_seconds": recorder.t_mission,
                   "diagnostic_seconds": diagnostic_seconds,
                   "path_length_m": recorder.accum_path_length}


def save_checkpoint(gaussian_map, recorder, event, observations, records):
    recorder.save_map(gaussian_map, f"{event:03}")
    recorder.save_path()
    record = {"event": event, "observations": observations,
              "mission_seconds": recorder.t_mission, "path_length_m": recorder.accum_path_length}
    records.append(record)
    atomic_json(Path(recorder.save_dir) / "checkpoints.json", records)


def load_prefix(path, cfg, device, destination):
    import numpy as np
    import torch
    from mapping.gaussian_map import GaussianMap
    from mapping.voxel_map import VoxelMap
    from utils.common import MissionRecorder
    cache = torch.load(path, map_location="cpu")
    random.setstate(cache["rng"]["python"])
    np.random.set_state(cache["rng"]["numpy"])
    torch.random.set_rng_state(cache["rng"]["torch_cpu"])
    torch.cuda.set_rng_state_all(cache["rng"]["torch_cuda"])
    gaussian_map = GaussianMap(cfg.mapper.gaussian_map, device)
    gaussian_map.__dict__.update(transfer_tree(cache["gaussian"], device))
    gaussian_map.device = device
    voxel_map = VoxelMap(cfg.mapper.voxel_map, cache["bbox"], device)
    voxel_map.__dict__.update(transfer_tree(cache["voxel"], device))
    voxel_map.device = device
    recorder = MissionRecorder(str(destination), cfg.experiment)
    recorder.__dict__.update(copy.deepcopy(cache["recorder"]))
    recorder.save_dir = str(destination)
    return cache, gaussian_map, voxel_map, recorder


def generate_prefix(cfg, protocol, seed, simulator, device, directory, context, context_hash):
    import numpy as np
    import torch
    from mapping.gaussian_map import GaussianMap
    from mapping.voxel_map import VoxelMap
    from utils.common import MissionRecorder
    from .planner import DefectPlanner
    directory.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    gaussian_map = GaussianMap(cfg.mapper.gaussian_map, device)
    voxel_map = VoxelMap(cfg.mapper.voxel_map, simulator.bbox, device)
    recorder = MissionRecorder(str(directory), cfg.experiment)
    planner = DefectPlanner(cfg.planner, device, method="confidence_nooracle",
                            diagnostics_dir=directory / "diagnostics")
    steps, checkpoints = [], []
    wall_start = time.perf_counter()
    time_ticks = protocol.as_dict()["time_checkpoints_seconds"]
    next_tick = 0
    for event in range(1, protocol.prefix + 1):
        frame, stats = acquire_event(gaussian_map, voxel_map, planner, simulator, recorder, seed, event)
        steps.append(stats)
        if event == 1:
            save_initial_rgb(frame, directory / "initial_rgb.png")
        time_checkpoint = next_tick < len(time_ticks) and recorder.t_mission >= time_ticks[next_tick]
        if event in protocol.checkpoints or time_checkpoint:
            save_checkpoint(gaussian_map, recorder, event, event, checkpoints)
            while next_tick < len(time_ticks) and recorder.t_mission >= time_ticks[next_tick]:
                next_tick += 1
    elapsed = time.perf_counter() - wall_start
    # The actual tensor state and raw RGB-D frames are persisted, not regenerated
    # from a seed. GaussianMap.save alone does not contain training_data.
    cache = {"gaussian": object_state(gaussian_map, ("optimizer",)),
             "voxel": object_state(voxel_map), "bbox": simulator.bbox,
             "recorder": object_state(recorder, ("save_dir",)),
             "planner_pose": planner.pose.cpu(), "planner_init": planner.init,
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                     "torch_cpu": torch.random.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()},
             "steps": steps, "checkpoints": checkpoints, "wall_seconds": elapsed}
    cache_path = directory / "prefix.th"
    torch.save(cache, cache_path)
    metadata = {"seed": seed, "observations": protocol.prefix,
                "cache_sha256": sha256_file(cache_path), "protocol": protocol.as_dict(),
                "context_hash": context_hash, "context": context,
                "mission_seconds": recorder.t_mission, "wall_seconds": elapsed,
                "peak_torch_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                "camera_sha256": hashlib.sha256(json.dumps(recorder.camera_params_list).encode()).hexdigest()}
    atomic_json(directory / "prefix.json", metadata)
    atomic_json(directory / "steps.json", steps)
    del gaussian_map, voxel_map, planner, recorder, cache
    torch.cuda.empty_cache()
    return cache_path, metadata


def run_branch(cfg, protocol, seed, method, simulator, device, prefix_path, prefix_meta, destination,
               versions):
    import shutil
    import torch
    from omegaconf import OmegaConf
    from .planner import DefectPlanner
    destination.mkdir(parents=True, exist_ok=False)
    cache, gaussian_map, voxel_map, recorder = load_prefix(prefix_path, cfg, device, destination)
    torch.cuda.reset_peak_memory_stats()
    planner = DefectPlanner(cfg.planner, device,
                            method="confidence_nooracle" if method == "refine_only" else method,
                            diagnostics_dir=destination / "diagnostics")
    planner.pose = cache["planner_pose"].cpu().clone()
    planner.init = cache["planner_init"]
    checkpoints = copy.deepcopy(cache["checkpoints"])
    steps = copy.deepcopy(cache["steps"])
    (destination / "map").mkdir()
    # Copy every prefix checkpoint; each branch starts at exactly these tensors.
    for source in (prefix_path.parent / "map").iterdir():
        shutil.copyfile(source, destination / "map" / source.name)
    shutil.copyfile(prefix_path.parent / "initial_rgb.png", destination / "initial_rgb.png")
    cfg_dump = OmegaConf.to_container(cfg, resolve=True)
    cfg_dump["planner"]["planner_name"] = method
    cfg_dump["experiment"].update(output_dir=str(destination.parents[4]), exp_id="benchmark", run_id=seed)
    atomic_json(destination / "exp_config.json", cfg_dump)
    experiment_protocol = {"version": protocol.as_dict()["version"], "protocol": protocol.as_dict(),
                           "seed": seed, "method": method, "source_versions": versions,
                           "scene": simulator.scene_name,
                           "context_hash": prefix_meta["context_hash"],
                           "scene_mesh_sha256": prefix_meta["context"]["scene_mesh_sha256"],
                           "eval_seed": domain_seed(0, "evaluation", 0),
                           "prefix_sha256": prefix_meta["cache_sha256"],
                           "prefix_camera_sha256": prefix_meta["camera_sha256"],
                           "refinement_post_processing": False if method == "refine_only" else True}
    atomic_json(destination / "protocol.json", experiment_protocol)
    atomic_json(destination / "checkpoints.json", checkpoints)
    if protocol.mode == "time" and recorder.t_mission >= protocol.seconds:
        raise RuntimeError("Shared prefix exhausted the time budget; use a shorter common prefix")
    branch_start = time.perf_counter()
    event = protocol.prefix
    simulator_calls_start = simulator.observation_calls
    time_ticks = protocol.as_dict()["time_checkpoints_seconds"]
    next_tick = next((index for index, tick in enumerate(time_ticks) if tick > recorder.t_mission), len(time_ticks))
    for event in range(protocol.prefix + 1, protocol.event_cap + 1):
        if protocol.mode == "time" and recorder.t_mission >= protocol.seconds:
            event -= 1
            break
        if method == "refine_only":
            with random_domain(seed, "mapping", event), gradient_refinement_only(gaussian_map):
                _, elapsed = synchronized_time(lambda: gaussian_map.train(steps=10))
            recorder.update_time("mapping", elapsed)
            stats = {"event": event, "observations": len(gaussian_map.training_data),
                     "planning_seconds": 0.0, "mapping_seconds": elapsed, "sensor_seconds": 0.0,
                     "diagnostic_seconds": 0.0,
                     "mission_seconds": recorder.t_mission, "path_length_m": recorder.accum_path_length}
        else:
            _, stats = acquire_event(gaussian_map, voxel_map, planner, simulator, recorder, seed, event)
        steps.append(stats)
        time_checkpoint = next_tick < len(time_ticks) and recorder.t_mission >= time_ticks[next_tick]
        if event in protocol.checkpoints or time_checkpoint:
            save_checkpoint(gaussian_map, recorder, event, len(gaussian_map.training_data), checkpoints)
            while next_tick < len(time_ticks) and recorder.t_mission >= time_ticks[next_tick]:
                next_tick += 1
        atomic_json(destination / "steps.json", steps)
    # A time-limited mission also saves its actual last map, not an earlier tick.
    if not checkpoints or checkpoints[-1]["event"] != event:
        save_checkpoint(gaussian_map, recorder, event, len(gaussian_map.training_data), checkpoints)
    recorder.save_path()
    summary = {"method": method, "seed": seed, "events": event,
               "observations": len(gaussian_map.training_data),
               "optimizer_steps": event * 10,
               "additional_optimizer_steps": (event - protocol.prefix) * 10,
               "mission_seconds": recorder.t_mission,
               "wall_seconds": cache["wall_seconds"] + time.perf_counter() - branch_start,
               "wall_time_scope": "reconstruction only; shared prefix and branch including diagnostic/checkpoint writes; excludes startup/prefix cache serialization and loading/meshing/evaluation",
               "shared_prefix_wall_seconds": cache["wall_seconds"],
               "sensor_seconds": sum(item["sensor_seconds"] for item in steps),
               "diagnostic_seconds": sum(item.get("diagnostic_seconds", 0.0) for item in steps),
               "peak_torch_allocated_mb": max(prefix_meta["peak_torch_allocated_mb"],
                                                torch.cuda.max_memory_allocated() / 1024 ** 2),
               "gpu_model": torch.cuda.get_device_name(),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "torch_visible_device_index": 0,
               "memory_scope": "PyTorch allocator only; excludes Habitat/OpenGL allocations",
               "mapping_seconds": recorder.t_mapping, "planning_seconds": recorder.t_planning,
               "simulated_flight_seconds": recorder.t_flight,
               "path_length_m": recorder.accum_path_length,
               "branch_observation_calls": simulator.observation_calls - simulator_calls_start,
               "blocked_future_observation_calls": simulator.blocked_calls,
               "prefix_sha256": prefix_meta["cache_sha256"],
               "checkpoint_count": len(checkpoints),
               "stop_reason": "time_budget" if protocol.mode == "time" and recorder.t_mission >= protocol.seconds
                              else "safety_event_cap" if protocol.mode == "time" else "event_limit"}
    if protocol.mode == "time" and summary["stop_reason"] == "safety_event_cap":
        raise RuntimeError("Time benchmark reached its safety event cap before its budget")
    atomic_json(destination / "run-summary.json", summary)
    experiment_protocol["cost"] = summary
    atomic_json(destination / "protocol.json", experiment_protocol)
    del gaussian_map, voxel_map, planner, recorder, cache
    torch.cuda.empty_cache()
    return summary


def evaluate_experiment(destination, ground_truth, sample_points):
    """Use the original metric definitions with explicit surface-sampling seeds."""
    import numpy as np
    import open3d as o3d
    from scipy.spatial import cKDTree
    import trimesh
    import torch
    from mapping.gaussian_map import GaussianMap
    from mesh_generation import generate_mesh
    from utils.operations import open3dmesh_2_trimesh
    evaluation_start = time.perf_counter()
    records = json.loads((destination / "checkpoints.json").read_text(encoding="utf-8"))
    ground_truth_seed = domain_seed(0, "evaluation", 0)
    gt_points, _ = trimesh.sample.sample_surface(ground_truth, sample_points, seed=ground_truth_seed)
    gt_tree = cKDTree(gt_points)
    output = {name: [] for name in ("step", "time", "path_length", "observation_count", "update_event",
                                   "mesh_accuracy", "mesh_completion", "mesh_completion_ratio",
                                   "mesh_chamfer_distance")}
    for record in records:
        event = record["event"]
        print(json.dumps({"stage": "meshing_evaluation", "experiment": str(destination),
                          "event": event, "observations": record["observations"]}), flush=True)
        gaussian_map = GaussianMap(None, torch.device("cuda"))
        gaussian_map.load(str(destination / "map" / f"map_{event:03}.th"))
        with (destination / "map" / f"cameras_{event:03}.pkl").open("rb") as stream:
            cameras = pickle.load(stream)
        mesh = generate_mesh(gaussian_map, cameras)
        mesh_path = destination / "map" / f"mesh_{event:03}.ply"
        if len(mesh.triangles) == 0 or not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
            raise RuntimeError(f"Empty or unwritable reconstruction mesh at event {event}")
        rec_points, _ = trimesh.sample.sample_surface(open3dmesh_2_trimesh(mesh), sample_points,
                                                     seed=domain_seed(0, "evaluation", event))
        if not np.isfinite(rec_points).all():
            raise RuntimeError("Non-finite reconstructed surface samples")
        accuracy = gt_tree.query(rec_points)[0].mean()
        completion_distances = cKDTree(rec_points).query(gt_points)[0]
        completion = completion_distances.mean()
        output["step"].append(event)
        output["update_event"].append(event)
        output["time"].append(record["mission_seconds"])
        output["path_length"].append(record["path_length_m"])
        output["observation_count"].append(record["observations"])
        output["mesh_accuracy"].append(float(accuracy * 100))
        output["mesh_completion"].append(float(completion * 100))
        output["mesh_completion_ratio"].append(float((completion_distances < 0.02).mean() * 100))
        output["mesh_chamfer_distance"].append(float((accuracy + completion) / 2))
        del gaussian_map, mesh, rec_points
        torch.cuda.empty_cache()
    output["evaluation"] = {"points_per_surface": sample_points, "threshold_m": 0.02,
                            "ground_truth_seed": ground_truth_seed,
                            "reconstruction_seed_rule": "domain_seed(0, evaluation, checkpoint_event)",
                            "sampling": "area-weighted surface; explicit trimesh seed",
                            "accuracy_unit": "cm", "completion_unit": "cm", "chamfer_unit": "m"}
    output["cost"] = json.loads((destination / "run-summary.json").read_text(encoding="utf-8"))
    output["cost"]["meshing_evaluation_wall_seconds"] = time.perf_counter() - evaluation_start
    atomic_json(destination / "final_result.json", output)
    return output


def validate_artifacts(destination):
    import importlib.util
    check_path = Path(__file__).resolve().parents[2] / "scripts/activegs/check_artifacts.py"
    specification = importlib.util.spec_from_file_location("viewmend_artifact_check", check_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    report = module.check(destination)
    results = json.loads((destination / "final_result.json").read_text(encoding="utf-8"))
    for event, count in zip(results["step"], results["observation_count"]):
        with (destination / "map" / f"cameras_{event:03}.pkl").open("rb") as stream:
            cameras = pickle.load(stream)
        if len(cameras) != count:
            raise ValueError("Checkpoint camera count disagrees with actual observation count")
    report["scope"] = "controlled run execution and map/camera/mesh/metric chain validation"
    report["observation_count"] = results["observation_count"]
    atomic_json(destination / "artifact-check.json", report)
    return report


def run_benchmark(upstream, run_dir, gpu, protocol, methods, seeds, scene="replica/office0",
                  prefix_cache_dir=None):
    import sys
    upstream, run_dir = Path(upstream).resolve(), Path(run_dir).resolve()
    if run_dir.exists():
        raise ValueError("run-dir must be new; existing experiments are never overwritten")
    admission = require_idle_gpu(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    previous_cwd = Path.cwd()
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    runtime = None
    backend = None
    pipeline_start = time.perf_counter()
    try:
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from simulator import get_simulator
        if not torch.cuda.is_available():
            raise RuntimeError("This benchmark requires the configured NVIDIA environment")
        device = torch.device("cuda")
        torch.cuda.init()
        with initialize_config_dir(config_dir=str(upstream / "config"), version_base=None):
            cfg = compose(config_name="main", overrides=["planner=confidence", f"scene={scene}",
                                                         "use_gui=false", "debug=false"])
        cfg.planner.sample_num = protocol.candidate_count
        cfg.planner.max_roi_sample_num = protocol.roi_count
        cfg.mapper.gaussian_map.optimization_steps = protocol.as_dict()["mapping_optimizer_steps_per_event"]
        cfg.scene.has_missing_surface = False
        cfg.experiment.record_rgbd = False
        cfg.experiment.record_global_path = True
        cfg.experiment.budget = protocol.seconds
        versions = source_versions(upstream)
        context, context_hash = prefix_context(cfg, protocol, versions)
        run_dir.mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "evidence").mkdir()
        runtime = {"protocol": protocol.as_dict(), "methods": list(methods), "seeds": list(seeds),
                   "gpu_admission": admission, "scene": scene, "torch": torch.__version__,
                   "requested_physical_gpu": gpu, "cuda_visible_devices": str(gpu),
                   "torch_visible_device_index": 0,
                   "source_versions": versions,
                   "config": OmegaConf.to_container(cfg, resolve=True),
                   "determinism": "Separated seeded RNGs; custom CUDA kernels may have numerical nondeterminism",
                   "status": "running", "experiments": []}
        atomic_json(run_dir / "benchmark-summary.json", runtime)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, check=True).stdout
        (run_dir / "evidence/pip-freeze.txt").write_text(freeze, encoding="utf-8")
        backend = get_simulator(cfg)
        simulator = ObservationOnlySimulator(backend)
        for seed in seeds:
            print(json.dumps({"stage": "common_prefix", "seed": seed, "frames": protocol.prefix}), flush=True)
            prefix_dir = Path(prefix_cache_dir) / f"seed_{seed}" if prefix_cache_dir else run_dir / "common-prefix" / f"seed_{seed}"
            if prefix_cache_dir:
                prefix_path = prefix_dir / "prefix.th"
                prefix_meta = json.loads((prefix_dir / "prefix.json").read_text(encoding="utf-8"))
                if prefix_meta["seed"] != seed or prefix_meta["protocol"] != protocol.as_dict():
                    raise ValueError("Reused common prefix has a different seed or protocol")
                if prefix_meta.get("context_hash") != context_hash:
                    raise ValueError("Reused common prefix has different configuration, source, or environment")
                if sha256_file(prefix_path) != prefix_meta["cache_sha256"]:
                    raise ValueError("Reused common-prefix cache checksum mismatch")
            else:
                prefix_path, prefix_meta = generate_prefix(cfg, protocol, seed, simulator, device, prefix_dir,
                                                         context, context_hash)
            for method in methods:
                destination = run_dir / "experiments" / "benchmark" / scene / method / str(seed)
                runtime["current_experiment"] = {"method": method, "seed": seed,
                                                 "experiment": str(destination.relative_to(run_dir)),
                                                 "stage": "reconstruction"}
                atomic_json(run_dir / "benchmark-summary.json", runtime)
                print(json.dumps(runtime["current_experiment"]), flush=True)
                summary = run_branch(cfg, protocol, seed, method, simulator, device,
                                     prefix_path, prefix_meta, destination, versions)
                entry = {**summary, "experiment": str(destination.relative_to(run_dir)), "status": "evaluating"}
                runtime["experiments"].append(entry)
                runtime["current_experiment"]["stage"] = "meshing_evaluation"
                atomic_json(run_dir / "benchmark-summary.json", runtime)
                evaluate_experiment(destination, backend.mesh, protocol.sample_points)
                validate_artifacts(destination)
                entry["status"] = "completed"
                atomic_json(run_dir / "benchmark-summary.json", runtime)
        runtime["status"] = "completed"
        runtime.pop("current_experiment", None)
        runtime["pipeline_wall_seconds"] = time.perf_counter() - pipeline_start
        runtime["future_observation_calls"] = simulator.blocked_calls
        atomic_json(run_dir / "benchmark-summary.json", runtime)
        return runtime
    except Exception as error:
        if runtime is not None:
            runtime.update(status="failed", error=f"{type(error).__name__}: {error}")
            if runtime["experiments"] and runtime["experiments"][-1]["status"] != "completed":
                runtime["experiments"][-1]["status"] = "failed"
            atomic_json(run_dir / "benchmark-summary.json", runtime)
        raise
    finally:
        if backend is not None:
            backend.sim.close()
        os.chdir(previous_cwd)
