"""Validate and publish recorded sensor inputs alongside their actual decisions."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_array(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Replay requires finite 4x4 camera-to-world poses")
    return pose


def read_decision(run, experiment, event, protocol, pose):
    source = experiment / "diagnostics" / f"step_{event:04d}.json"
    if event <= protocol["protocol"]["prefix"]:
        source = run / "common-prefix" / f"seed_{protocol['seed']}" / "diagnostics" / source.name
    decision = json.loads(source.read_text(encoding="utf-8"))
    if decision.get("step") != event or not np.allclose(pose_array(decision["selected_pose"]), pose, atol=1e-5, rtol=0):
        raise ValueError("Decision event/selected pose does not match the acquired frame")
    poses = decision.get("candidate_poses", [])
    selected = decision.get("selected_index")
    if decision.get("initialization"):
        if poses or selected is not None or event != 1:
            raise ValueError("Invalid initial acquisition decision")
    else:
        if type(selected) is not int or not 0 <= selected < len(poses):
            raise ValueError("Invalid chosen candidate index")
        for item in poses:
            pose_array(item)
        if not np.allclose(pose_array(poses[selected]), pose, atol=1e-5, rtol=0):
            raise ValueError("Chosen candidate differs from actual sensor pose")
        for field in ("reachable", "final_scores", "exploration", "uncertainty", "defect", "path_lengths",
                      "baseline_scores", "geometry_bonus"):
            values = decision.get(field)
            if not isinstance(values, list) or len(values) != len(poses):
                raise ValueError(f"Missing or misaligned candidate {field}")
        if decision["reachable"][selected] is not True or not np.isfinite(decision["final_scores"][selected]):
            raise ValueError("Chosen candidate must be reachable with a finite score")
        baseline = decision.get("baseline_selected_index")
        if type(baseline) is not int or not 0 <= baseline < len(poses):
            raise ValueError("Invalid recorded baseline candidate")
        reachable = [i for i, value in enumerate(decision["reachable"]) if value is True]
        if any(not np.isfinite(decision["final_scores"][i]) for i in reachable):
            raise ValueError("Reachable candidates need finite scores")
        if selected != max(reachable, key=lambda i: decision["final_scores"][i]):
            raise ValueError("Selected candidate does not maximize recorded score")
    if decision.get("future_candidate_observation_queries") != 0:
        raise ValueError("Replay planner queried future sensor depth")
    fields = ("initialization", "method", "candidate_poses", "selected_index", "baseline_selected_index",
              "reachable", "exploration", "uncertainty", "defect", "path_lengths", "final_scores",
              "baseline_scores", "geometry_bonus", "geometry_bonus_cap", "geometry_active",
              "geometry_fallback_reason", "selection_changed", "all_zero_utility_fallback", "baseline_regret")
    return {**{field: decision.get(field) for field in fields}, "source_sha256": digest(source)}


def export_replay(run, experiment, output, manifest):
    folder = experiment / "observations"
    enabled = manifest["protocol"].get("replay_recording")
    if not enabled:
        if folder.exists():
            raise ValueError("Recorded frames lack an explicit replay protocol")
        return None
    protocol = manifest["protocol"]
    definitions = protocol["protocol"]
    if (definitions.get("observations") != 8 or definitions.get("prefix") != 1
            or definitions.get("checkpoint_every") != 1 or manifest["method_id"] != "defect_guarded"
            or len(manifest["cameras"]) != 8 or len(manifest["checkpoints"]) != 8):
        raise ValueError("Replay must be the independent eight-frame/eight-map demo")
    metadata_files = sorted(folder.glob("frame_*.json"))
    if len(metadata_files) != 8:
        raise ValueError("Replay cannot publish an incomplete acquired frame sequence")
    frames = []
    destination = output / "replay"
    destination.mkdir(parents=True, exist_ok=True)
    for index, file in enumerate(metadata_files):
        record = json.loads(file.read_text(encoding="utf-8"))
        event = index + 1
        if (record.get("event") != event or record.get("schema") != "viewmend-acquired-observation-v1"
                or record.get("source") != "selected_view_sensor_before_mapping"):
            raise ValueError("Invalid actual observation record")
        pose = pose_array(record["pose"])
        camera = manifest["cameras"][index]
        if (not np.allclose(pose[:3, 3], camera["position"], atol=1e-5, rtol=0)
                or not np.allclose(pose[:3, :3].reshape(-1), camera["rotation"], atol=1e-5, rtol=0)):
            raise ValueError("Recorded frame pose differs from saved acquisition camera")
        checkpoint = manifest["checkpoints"][index]
        if checkpoint["observation_count"] != event or checkpoint["update_event"] != event:
            raise ValueError("Replay maps do not align one-to-one with acquisitions")
        sources = {}
        for key in ("rgb", "depth", "depth_raw"):
            name = record[key]
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("Observation assets must be confined filenames")
            path = folder / name
            path.resolve().relative_to(folder.resolve())
            if digest(path) != record["sha256"][key]:
                raise ValueError("Recorded observation hash mismatch")
            sources[key] = path
        raw = np.load(sources["depth_raw"], allow_pickle=False)
        if raw.shape != (record["height"], record["width"]) or raw.dtype != np.float32:
            raise ValueError("Invalid raw metric depth shape/type")
        near, far = record["depth_range_m"]
        if not np.isfinite([near, far]).all() or not 0 <= near < far:
            raise ValueError("Invalid fixed metric depth legend")
        valid = np.isfinite(raw) & (raw > near) & (raw <= far)
        if int(valid.sum()) != record["valid_depth_pixels"]:
            raise ValueError("Depth metadata does not describe raw acquired depth")
        for key in ("rgb", "depth"):
            with Image.open(sources[key]) as image:
                if image.size != (record["width"], record["height"]):
                    raise ValueError("Recorded image dimensions do not match")
            shutil.copyfile(sources[key], destination / record[key])
        decision = read_decision(run, experiment, event, protocol, pose)
        frames.append({**{key: record[key] for key in ("event", "pose", "intrinsic", "width", "height",
                       "depth_range_m", "valid_depth_pixels", "sha256")},
                       "rgb": "replay/" + record["rgb"], "depth": "replay/" + record["depth"],
                       "checkpoint_index": index, "decision": decision,
                       "observation_metadata_sha256": digest(file)})
    replay = {"schema": "viewmend-capture-replay-v1", "recorded_sensor_only": True,
              "scope": "independent_visual_demo_not_formal_quality_sample", "frames": frames,
              "decision_alignment": "frame[i].decision selects frame[i]; current frame i previews decision of frame i+1",
              "camera_convention": manifest["camera_convention"], "depth_unit": "m"}
    target = output / "replay.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(replay, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(target)
    return {"file": "replay.json", "sha256": digest(target), "frame_count": len(frames),
            "scope": replay["scope"]}
