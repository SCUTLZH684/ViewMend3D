"""Audit current ActiveGS files with the Python standard library.

Camera pickles contain numeric lists written by MissionRecorder. Tensor ZIPs
are inspected without loading tensors. This does not validate evaluator logic.
"""
import argparse
import hashlib
import json
import math
import pickle
import struct
from pathlib import Path
import zipfile

METRICS = (
    "mesh_accuracy", "mesh_completion", "mesh_completion_ratio",
    "mesh_chamfer_distance",
)
PLY_TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid finite nonnegative {name}")
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ply_counts(path):
    """Inspect complete mesh payload, finite properties and face indices."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Invalid PLY file: {path.name}")
        elements, encoding = [], None
        for _ in range(100):
            raw = stream.readline()
            if not raw:
                raise ValueError(f"Truncated PLY header: {path.name}")
            line = raw.decode("ascii").strip().split()
            if not line or line[0] in ("comment", "obj_info"):
                continue
            if line[0] == "format":
                if len(line) != 3 or line[2] != "1.0":
                    raise ValueError("Unsupported PLY format/version")
                encoding = line[1]
            elif line[0] == "element":
                if len(line) != 3 or int(line[2]) < 0:
                    raise ValueError("Invalid PLY element")
                elements.append((line[1], int(line[2]), []))
            elif line[0] == "property":
                if not elements:
                    raise ValueError("PLY property precedes element")
                if len(line) == 3 and line[1] in PLY_TYPES:
                    elements[-1][2].append((line[2], line[1], None))
                elif len(line) == 5 and line[1] == "list" and line[2] in PLY_TYPES and line[3] in PLY_TYPES:
                    if PLY_TYPES[line[2]] in "fd":
                        raise ValueError("PLY list count must be an integer")
                    elements[-1][2].append((line[4], line[3], line[2]))
                else:
                    raise ValueError("Unsupported PLY property")
            elif line[0] == "end_header":
                break
        else:
            raise ValueError("PLY header exceeds inspection limit")
        counts = {name: count for name, count, _ in elements}
        if len(counts) != len(elements) or counts.get("vertex", 0) <= 0 or counts.get("face", 0) <= 0:
            raise ValueError(f"Empty/duplicate reconstructed mesh elements: {path.name}")
        props_by_element = {name: props for name, _, props in elements}
        if not {"x", "y", "z"} <= {name for name, _, count_type in props_by_element["vertex"] if count_type is None}:
            raise ValueError("Mesh vertices require XYZ coordinates")
        if not any(name in ("vertex_indices", "vertex_index") and count_type and PLY_TYPES[value_type] not in "fd"
                   for name, value_type, count_type in props_by_element["face"]):
            raise ValueError("Mesh faces require integer vertex indices")
        if encoding not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError("Unsupported PLY encoding")
        endian = "<" if encoding == "binary_little_endian" else ">"
        parsers = {kind: struct.Struct(endian + code) for kind, code in PLY_TYPES.items()}
        for element, count, props in elements:
            for _ in range(count):
                tokens = iter(stream.readline().decode("ascii").split()) if encoding == "ascii" else None

                def scalar(kind):
                    if tokens is None:
                        parser = parsers[kind]
                        payload = stream.read(parser.size)
                        if len(payload) != parser.size:
                            raise ValueError(f"Truncated PLY payload: {path.name}")
                        return parser.unpack(payload)[0]
                    try:
                        token = next(tokens)
                    except StopIteration as error:
                        raise ValueError(f"Truncated PLY row: {path.name}") from error
                    return float(token) if PLY_TYPES[kind] in "fd" else int(token)

                for name, value_type, count_type in props:
                    size = scalar(count_type) if count_type else 1
                    if count_type and not 0 <= size <= counts["vertex"]:
                        raise ValueError("Invalid PLY list length")
                    values = [scalar(value_type) for _ in range(size)]
                    if PLY_TYPES[value_type] in "fd" and any(not math.isfinite(value) for value in values):
                        raise ValueError("Non-finite PLY property")
                    if element == "face" and name in ("vertex_indices", "vertex_index"):
                        if size < 3 or any(not 0 <= value < counts["vertex"] for value in values):
                            raise ValueError("Invalid mesh face indices")
                if tokens is not None and next(tokens, None) is not None:
                    raise ValueError("Unexpected values in PLY row")
        if stream.read().strip():
            raise ValueError("Unexpected trailing PLY payload")
    return counts


class CameraUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Camera pickle must contain plain numeric lists")


def read_cameras(path):
    with path.open("rb") as stream:
        cameras = CameraUnpickler(stream).load()
        if stream.read():
            raise ValueError("Unexpected trailing camera pickle data")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("Acquisition cameras must be a nonempty list")
    for row in cameras:
        if not isinstance(row, list) or len(row) != 25 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in row):
            raise ValueError("Camera parameters require 16 finite extrinsic and 9 intrinsic values")
        if any(abs(value - expected) > 1e-4 for value, expected in zip(row[12:16], (0, 0, 0, 1))):
            raise ValueError("Invalid camera-to-world homogeneous transform")
        if row[16] <= 0 or row[20] <= 0 or any(abs(value - expected) > 1e-4 for value, expected in zip(row[22:25], (0, 0, 1))):
            raise ValueError("Invalid camera intrinsic matrix")
        rotation = [[row[r * 4 + c] for c in range(3)] for r in range(3)]
        for r in range(3):
            for c in range(3):
                if abs(sum(rotation[k][r] * rotation[k][c] for k in range(3)) - (r == c)) > 1e-4:
                    raise ValueError("Camera rotation must be orthonormal")
        determinant = sum(rotation[0][c] * (rotation[1][(c + 1) % 3] * rotation[2][(c + 2) % 3]
                          - rotation[1][(c + 2) % 3] * rotation[2][(c + 1) % 3]) for c in range(3))
        if abs(determinant - 1) > 1e-4:
            raise ValueError("Camera rotation must preserve orientation")
    return cameras


def map_container(path):
    """Check Torch ZIP integrity; tensor semantics remain unverified here."""
    if not zipfile.is_zipfile(path):
        raise ValueError("Map must be a complete Torch ZIP checkpoint")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not any(name.endswith("/data.pkl") for name in names):
            raise ValueError("Invalid Torch map container entries")
        if not any("/data/" in name and archive.getinfo(name).file_size > 0 for name in names):
            raise ValueError("Map container contains no tensor storage")
        if archive.testzip() is not None:
            raise ValueError("Corrupt map tensor storage")


def check(experiment):
    experiment = Path(experiment)
    record = experiment / "map" / "record_info.txt"
    rows = [list(map(float, line.split())) for line in record.read_text().splitlines() if line.strip()]
    if len(rows) < 2 or any(len(row) != 3 for row in rows):
        raise ValueError("Upstream mesh/eval loaders require at least two checkpoint rows")
    for row in rows:
        for value in row:
            finite_nonnegative(value, "checkpoint record")
    checkpoint_ids = [int(row[0]) for row in rows]
    if any(row[0] != checkpoint_id or checkpoint_id <= 0 for row, checkpoint_id in zip(rows, checkpoint_ids)):
        raise ValueError("Checkpoint IDs must be positive integers")
    if any(b <= a for a, b in zip(checkpoint_ids, checkpoint_ids[1:])):
        raise ValueError("Checkpoint IDs must be ordered and unique")
    for column in (1, 2):
        if any(b[column] < a[column] for a, b in zip(rows, rows[1:])):
            raise ValueError("Recorded mission time/path cannot decrease")
    results = json.loads((experiment / "final_result.json").read_text(encoding="utf-8"))
    for name in METRICS + ("step", "time", "path_length"):
        values = results.get(name)
        if not isinstance(values, list) or len(values) != len(rows):
            raise ValueError(f"Metric length mismatch: {name}")
        for value in values:
            finite_nonnegative(value, name)
    if results["step"] != checkpoint_ids or any(value > 100 for value in results["mesh_completion_ratio"]):
        raise ValueError("Invalid steps or completion percentages")
    for field, column in (("time", 1), ("path_length", 2)):
        if any(not math.isclose(value, row[column], rel_tol=1e-9, abs_tol=1e-8) for value, row in zip(results[field], rows)):
            raise ValueError(f"Checkpoint record and result {field} disagree")
    for accuracy_cm, completion_cm, chamfer_m in zip(results["mesh_accuracy"], results["mesh_completion"], results["mesh_chamfer_distance"]):
        if not math.isclose(chamfer_m, (accuracy_cm + completion_cm) / 200, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("Chamfer is inconsistent with the upstream distance definition/units")
    for field in ("observation_count", "update_event"):
        if field in results and (not isinstance(results[field], list) or len(results[field]) != len(rows)
                               or any(type(value) is not int or value <= 0 for value in results[field])):
            raise ValueError(f"Invalid integer checkpoint field: {field}")
    if "update_event" in results and results["update_event"] != checkpoint_ids:
        raise ValueError("Update events and checkpoint IDs disagree")
    artifacts, camera_counts, files, previous = [], [], [record, experiment / "final_result.json"], []
    for index, checkpoint_id in enumerate(checkpoint_ids):
        paths = [experiment / "map" / filename for filename in
                 (f"map_{checkpoint_id:03}.th", f"cameras_{checkpoint_id:03}.pkl", f"mesh_{checkpoint_id:03}.ply")]
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing/empty artifact: {path.name}")
        map_container(paths[0])
        cameras = read_cameras(paths[1])
        if cameras[:len(previous)] != previous:
            raise ValueError("Camera checkpoints must preserve the exact acquisition prefix")
        if len(cameras) > checkpoint_id:
            raise ValueError("Camera count cannot exceed update events")
        if "observation_count" in results and len(cameras) != results["observation_count"][index]:
            raise ValueError("Checkpoint camera count disagrees with actual observation count")
        camera_counts.append(len(cameras))
        previous = cameras
        artifacts.append({"checkpoint": checkpoint_id, "mesh": ply_counts(paths[2]), "camera_count": len(cameras)})
        files.extend(paths)
    traveled = sum(math.dist(a[3:12:4], b[3:12:4]) for a, b in zip(previous, previous[1:]))
    if traveled > rows[-1][2] + 1e-3:
        raise ValueError("Acquisition camera displacement exceeds recorded path length")
    path_file = experiment / "global_path.pkl"
    if path_file.is_file():
        files.append(path_file)
    fingerprints = {path.relative_to(experiment).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files}
    return {"artifact_chain_passed": True, "checkpoint_count": len(rows),
            "artifacts": artifacts, "observation_count": camera_counts, "files": fingerprints,
            "last_checkpoint_metrics": {name: results[name][-1] for name in METRICS},
            "validation_version": "stdlib-current-artifacts-v2",
            "scope": "current file integrity, camera prefixes and checkpoint/metric alignment; tensor contents, global-path poses and evaluator logic are not decoded by this standard-library check"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.experiment)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
