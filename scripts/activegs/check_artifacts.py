"""Check the official ActiveGS output chain without importing GPU libraries."""
import argparse
import json
import math
from pathlib import Path

METRICS = (
    "mesh_accuracy", "mesh_completion", "mesh_completion_ratio",
    "mesh_chamfer_distance",
)


def ply_counts(path):
    counts = {}
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Invalid PLY file: {path.name}")
        for _ in range(100):
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element "):
                _, name, count = line.split()
                counts[name] = int(count)
            if line == "end_header":
                break
        else:
            raise ValueError("PLY header exceeds inspection limit")
    if counts.get("vertex", 0) <= 0 or counts.get("face", 0) <= 0:
        raise ValueError(f"Empty reconstructed mesh: {path.name}")
    return counts


def check(experiment):
    record = experiment / "map" / "record_info.txt"
    rows = [list(map(float, line.split())) for line in record.read_text().splitlines() if line.strip()]
    if len(rows) < 2 or any(len(row) != 3 for row in rows):
        raise ValueError("Upstream mesh/eval loaders require at least two checkpoint rows")
    checkpoint_ids = [int(row[0]) for row in rows]
    if any(row[0] != checkpoint_id for row, checkpoint_id in zip(rows, checkpoint_ids)):
        raise ValueError("Checkpoint IDs must be integers")
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("Duplicate checkpoint IDs")
    artifacts = []
    for checkpoint_id in checkpoint_ids:
        for filename in (f"map_{checkpoint_id:03}.th", f"cameras_{checkpoint_id:03}.pkl", f"mesh_{checkpoint_id:03}.ply"):
            path = experiment / "map" / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing/empty artifact: {filename}")
        mesh = experiment / "map" / f"mesh_{checkpoint_id:03}.ply"
        artifacts.append({"checkpoint": checkpoint_id, "mesh": ply_counts(mesh)})
    results = json.loads((experiment / "final_result.json").read_text())
    for name in METRICS + ("step", "time", "path_length"):
        values = results.get(name)
        if not isinstance(values, list) or len(values) != len(rows):
            raise ValueError(f"Metric length mismatch: {name}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Invalid metric value: {name}")
    if results["step"] != checkpoint_ids or any(value > 100 for value in results["mesh_completion_ratio"]):
        raise ValueError("Invalid steps or completion percentages")
    for accuracy_cm, completion_cm, chamfer_m in zip(results["mesh_accuracy"], results["mesh_completion"], results["mesh_chamfer_distance"]):
        if not math.isclose(chamfer_m, (accuracy_cm + completion_cm) / 200, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("Chamfer is inconsistent with the upstream distance definition/units")
    return {"artifact_chain_passed": True, "checkpoint_count": len(rows),
            "artifacts": artifacts, "last_checkpoint_metrics": {name: results[name][-1] for name in METRICS},
            "scope": "single-run execution and output validation; paper-level results not reproduced"}


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
