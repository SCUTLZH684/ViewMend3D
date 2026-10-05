"""Audit paired experiment outputs before producing descriptive statistics.

No GPU libraries are imported. Original/oracle runs, mixed protocols, duplicate
seeds and unequal common prefixes cannot silently enter the same comparison.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

VERSION = "viewmend-observed-only-v1"
METRICS = ("mesh_accuracy", "mesh_completion", "mesh_completion_ratio", "mesh_chamfer_distance")
COSTS = ("mission_seconds", "wall_seconds", "planning_seconds", "mapping_seconds",
         "sensor_seconds", "path_length_m", "observations", "optimizer_steps")
IDENTITY = ("scene", "scene_mesh_sha256", "context_hash")


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid finite nonnegative {name}")
    return value


def read_run(experiment):
    protocol = json.loads((experiment / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((experiment / "final_result.json").read_text(encoding="utf-8"))
    if protocol.get("version") != VERSION:
        raise ValueError(f"Unsupported/original protocol: {experiment}")
    seed = protocol.get("seed")
    if type(seed) is not int or seed < 0 or not isinstance(protocol.get("method"), str):
        raise ValueError("Method and RNG seed must be recorded explicitly")
    if not isinstance(protocol.get("scene"), str) or not protocol["scene"]:
        raise ValueError("Scene identity must be recorded explicitly")
    for key in ("scene_mesh_sha256", "context_hash"):
        digest = protocol.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"An exact scene/configuration digest is required: {key}")
    recipe = protocol["protocol"]
    if recipe.get("future_candidate_depth_mask") is not False:
        raise ValueError("Future candidate masks are forbidden in a fair comparison")
    n = len(result.get("step", []))
    if n < 2:
        raise ValueError("At least two complete checkpoints are required")
    for key in METRICS + ("step", "time", "path_length", "observation_count", "update_event"):
        values = result.get(key)
        if not isinstance(values, list) or len(values) != n:
            raise ValueError(f"Missing/misaligned checkpoint field: {key}")
        for value in values:
            number(value, key)
    if result["step"] != result["update_event"] or any(b <= a for a, b in zip(result["step"], result["step"][1:])):
        raise ValueError("Checkpoint events must be unique, ordered and correctly labelled")
    for accuracy, completion, coverage, chamfer in zip(*(result[key] for key in METRICS)):
        if coverage > 100 or not math.isclose(chamfer, (accuracy + completion) / 200, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("Invalid coverage or mismatched distance units")
    prefix = recipe["prefix"]
    if protocol["method"] == "refine_only":
        if any(value != prefix for value in result["observation_count"]):
            raise ValueError("The refinement control must not acquire new observations")
    elif result["observation_count"] != result["update_event"]:
        raise ValueError("Acquisition runs must distinguish actual observations from update events")
    if recipe["mode"] == "observations" and result["step"][-1] != recipe["observations"]:
        raise ValueError("Observation protocol did not finish the requested update events")
    cost = protocol.get("cost", {})
    for key in COSTS:
        number(cost.get(key), key)
    if cost["observations"] != result["observation_count"][-1]:
        raise ValueError("Cost and checkpoint observation counts disagree")
    prefix_hash = protocol.get("prefix_sha256")
    if not isinstance(prefix_hash, str) or len(prefix_hash) != 64:
        raise ValueError("An exact common-prefix cache digest is required")
    return {"method": protocol["method"], "seed": seed, "protocol": protocol,
            "result": result, "cost": cost, "experiment": experiment}


def stats(values):
    return {"mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def aggregate(experiments, methods, expected_seeds):
    runs = [read_run(path) for path in experiments]
    if not runs:
        raise ValueError("No completed controlled experiments found")
    pairs = {}
    for run in runs:
        key = run["method"], run["seed"]
        if key in pairs:
            raise ValueError(f"Duplicate method/seed: {key}")
        pairs[key] = run
    requested = {(method, seed) for method in methods for seed in expected_seeds}
    if set(pairs) != requested:
        raise ValueError(f"Incomplete/mixed suite; missing={sorted(requested - set(pairs))}; unexpected={sorted(set(pairs) - requested)}")
    recipe = runs[0]["protocol"]["protocol"]
    eval_seed = runs[0]["protocol"]["eval_seed"]
    evaluation = runs[0]["result"]["evaluation"]
    source = runs[0]["protocol"]["source_versions"]
    identity = {key: runs[0]["protocol"][key] for key in IDENTITY}
    for run in runs:
        current = run["protocol"]
        if current["protocol"] != recipe or current["eval_seed"] != eval_seed:
            raise ValueError("Mixed acquisition/optimization/randomness protocols")
        if run["result"]["evaluation"] != evaluation or current["source_versions"] != source:
            raise ValueError("Mixed evaluator or source versions")
        if {key: current[key] for key in IDENTITY} != identity:
            raise ValueError("Mixed scenes, scene assets or resolved configurations")
    for seed in expected_seeds:
        digests = {pairs[method, seed]["protocol"]["prefix_sha256"] for method in methods}
        cameras = {pairs[method, seed]["protocol"].get("prefix_camera_sha256") for method in methods}
        if len(digests) != 1 or len(cameras) != 1:
            raise ValueError(f"Unequal actual prefixes for seed {seed}")
    per_run = []
    by_method = {}
    for method in methods:
        selected = [pairs[method, seed] for seed in expected_seeds]
        by_method[method] = {key: stats([run["result"][key][-1] for run in selected]) for key in METRICS}
        by_method[method]["cost"] = {key: stats([run["cost"][key] for run in selected]) for key in COSTS}
        for run in selected:
            per_run.append({"method": method, "seed": run["seed"], "prefix_sha256": run["protocol"]["prefix_sha256"],
                            "metrics": {key: run["result"][key][-1] for key in METRICS}, "cost": run["cost"]})
    paired = {}
    if "confidence_nooracle" in methods:
        for method in methods:
            if method == "confidence_nooracle":
                continue
            paired[method] = {}
            for key in METRICS:
                differences = [pairs[method, seed]["result"][key][-1] - pairs["confidence_nooracle", seed]["result"][key][-1]
                               for seed in expected_seeds]
                higher_better = key == "mesh_completion_ratio"
                paired[method][key] = {**stats(differences), "differences_by_seed": dict(zip(map(str, expected_seeds), differences)),
                                      "improved_seeds": sum(value > 0 if higher_better else value < 0 for value in differences)}
    return {"version": VERSION, "protocol": recipe, "evaluation": evaluation, "source_versions": source,
            **identity,
            "seeds": expected_seeds, "methods": by_method, "per_run": per_run,
            "paired_difference_vs_confidence": paired,
            "scope": "Descriptive paired statistics on the recorded single scene; no statistical significance or cross-scene generalization claim."}


def markdown(report):
    lines = ["# ViewMend3D 配对实验结果", "", "各方法使用同种子的同一个前缀缓存。数值为均值 ± 样本标准差；单次结果不报告标准差。", "",
             "| 方法 | Accuracy cm ↓ | Completion cm ↓ | 覆盖率 % ↑ | Chamfer mm ↓ | 观测数 | 规划秒 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    def cell(item, scale=1):
        mean = item["mean"] * scale
        std = item["sample_std"]
        return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std * scale:.4f}"
    for method, metrics in report["methods"].items():
        lines.append(f"| {method} | {cell(metrics['mesh_accuracy'])} | {cell(metrics['mesh_completion'])} | {cell(metrics['mesh_completion_ratio'])} | {cell(metrics['mesh_chamfer_distance'], 1000)} | {cell(metrics['cost']['observations'])} | {cell(metrics['cost']['planning_seconds'])} |")
    lines += ["", f"结果范围为 {report['scene']} 的配对描述统计，不表示统计显著，也不能推广到其他场景。完整的逐种子数据、成本、来源版本与配对差值见同名 JSON。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--methods", nargs="+", default=["confidence_nooracle", "random_matched", "defect"])
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(path.parent for path in (args.run / "experiments/benchmark/replica/office0").glob("*/*/final_result.json")
                   if path.parent.parent.name in args.methods)
    report = aggregate(paths, args.methods, args.expected_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"audited_runs": len(report["per_run"]), "seeds": args.expected_seeds, "output": str(args.output)}, ensure_ascii=False))
