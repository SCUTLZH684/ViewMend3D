"""Optional records of actually acquired frames; never query a candidate sensor."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_observation(frame, directory, event):
    """Persist selected-view inputs before mapping; this is demo-only extra IO."""
    import numpy as np
    from PIL import Image

    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    if type(event) is not int or event <= 0:
        raise ValueError("Observation event must be a positive integer")
    rgb = array(frame["rgb"])
    depth = array(frame["depth"])
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    pose, intrinsic = array(frame["extrinsic"]), array(frame["intrinsic"])
    limits = array(frame["depth_range"]).reshape(-1)
    if (rgb.ndim != 3 or rgb.shape[0] != 3 or depth.shape != rgb.shape[1:]
            or pose.shape != (4, 4) or intrinsic.shape != (3, 3)
            or len(limits) != 2 or not np.isfinite(limits).all() or not 0 <= limits[0] < limits[1]
            or not np.isfinite(rgb).all() or not np.isfinite(pose).all() or not np.isfinite(intrinsic).all()):
        raise ValueError("Invalid acquired RGB-D/camera frame")
    root = Path(directory) / "observations"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{event:04d}"
    names = {"rgb": stem + "_rgb.png", "depth": stem + "_depth.png", "depth_raw": stem + "_depth.npy"}
    metadata = root / (stem + ".json")
    if metadata.exists() or any((root / name).exists() for name in names.values()):
        raise ValueError("Recorded observations must not be overwritten")
    pixels = (np.clip(rgb, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(root / names["rgb"])
    valid = np.isfinite(depth) & (depth > limits[0]) & (depth <= limits[1])
    scaled = np.clip((np.where(valid, depth, limits[0]) - limits[0]) / (limits[1] - limits[0]), 0, 1)
    palette = np.asarray([[35, 42, 94], [38, 94, 139], [35, 156, 152], [133, 205, 104], [250, 231, 128]])
    colored = np.stack([np.interp(scaled, np.linspace(0, 1, len(palette)), palette[:, channel]) for channel in range(3)], -1).astype(np.uint8)
    colored[~valid] = [52, 57, 65]
    Image.fromarray(colored).save(root / names["depth"])
    np.save(root / names["depth_raw"], depth.astype(np.float32), allow_pickle=False)
    record = {"schema": "viewmend-acquired-observation-v1", "event": event,
              "source": "selected_view_sensor_before_mapping", "width": int(depth.shape[1]),
              "height": int(depth.shape[0]), "pose": pose.tolist(), "intrinsic": intrinsic.tolist(),
              "depth_range_m": limits.tolist(), "valid_depth_pixels": int(valid.sum()),
              "invalid_depth_color": [52, 57, 65], "depth_palette": palette.tolist(),
              **names, "sha256": {key: sha256(root / name) for key, name in names.items()}}
    metadata.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    return record
