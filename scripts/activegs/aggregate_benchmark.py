"""Audit paired experiment outputs before producing descriptive statistics.

No GPU libraries are imported. Original/oracle runs, mixed protocols, duplicate
seeds and unequal common prefixes cannot silently enter the same comparison.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from check_artifacts import check as check_current_artifacts

VERSION = "viewmend-observed-only-v1"
METRICS = ("mesh_accuracy", "mesh_completion", "mesh_completion_ratio", "mesh_chamfer_distance")
COSTS = ("mission_seconds", "wall_seconds", "planning_seconds", "mapping_seconds",
         "sensor_seconds", "path_length_m", "observations", "optimizer_steps")
IDENTITY = ("scene", "scene_mesh_sha256", "context_hash")
METHODS = ("confidence_nooracle", "random_matched", "defect", "defect_no_gate", "refine_only")


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid finite nonnegative {name}")
    return value


def equal_number(actual, expected, name):
    number(actual, name)
    number(expected, name)
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-7):
        raise ValueError(f"Recorded {name} values disagree")


def digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"An exact digest is required: {name}")
    return value


def audit_execution(experiment, protocol, result):
    """Audit the event ledger and actual stopping condition, not a success flag."""
    recipe, cost = protocol["protocol"], protocol["cost"]
    steps = json.loads((experiment / "steps.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((experiment / "checkpoints.json").read_text(encoding="utf-8"))
    summary = json.loads((experiment / "run-summary.json").read_text(encoding="utf-8"))
    events = cost.get("events")
    if type(events) is not int or events < recipe["prefix"] or not isinstance(steps, list) or len(steps) != events:
        raise ValueError("Complete ordered event ledger is required")
    previous_mission = previous_path = 0.0
    for event, step in enumerate(steps, 1):
        if not isinstance(step, dict) or type(step.get("event")) is not int or step["event"] != event:
            raise ValueError("Event ledger must record each update exactly once")
        expected_observations = min(event, recipe["prefix"]) if protocol["method"] == "refine_only" else event
        if type(step.get("observations")) is not int or step["observations"] != expected_observations:
            raise ValueError("Event ledger observation counts disagree with the method")
        for name in ("mission_seconds", "path_length_m", "planning_seconds", "mapping_seconds", "sensor_seconds"):
            number(step.get(name), name)
        if step["mission_seconds"] < previous_mission or step["path_length_m"] < previous_path:
            raise ValueError("Event ledger mission time/path cannot decrease")
        increment = step["mission_seconds"] - previous_mission
        if increment + 1e-7 < step["planning_seconds"] + step["mapping_seconds"]:
            raise ValueError("Event mission time excludes recorded planning/mapping work")
        if protocol["method"] == "refine_only" and event > recipe["prefix"]:
            if step["planning_seconds"] != 0 or step["sensor_seconds"] != 0 or step["path_length_m"] != previous_path:
                raise ValueError("Refinement events must not plan, acquire sensors or move")
            equal_number(increment, step["mapping_seconds"], "refinement mission increment")
        previous_mission, previous_path = step["mission_seconds"], step["path_length_m"]
    if not isinstance(checkpoints, list) or len(checkpoints) != len(result["step"]):
        raise ValueError("Actual checkpoint ledger does not match evaluated checkpoints")
    for index, checkpoint in enumerate(checkpoints):
        event = result["step"][index]
        if (type(checkpoint.get("event")) is not int or checkpoint["event"] != event
                or type(checkpoint.get("observations")) is not int
                or checkpoint["observations"] != result["observation_count"][index]):
            raise ValueError("Checkpoint ledger does not match evaluated events/observations")
        for field, source in (("time", "mission_seconds"), ("path_length", "path_length_m")):
            equal_number(checkpoint.get(source), result[field][index], field)
            equal_number(checkpoint.get(source), steps[event - 1][source], field)
    if result["step"][-1] != events:
        raise ValueError("The evaluated final map must be the actual last update")
    for name in COSTS + ("events", "simulated_flight_seconds", "branch_observation_calls", "blocked_future_observation_calls", "checkpoint_count"):
        equal_number(summary.get(name), cost.get(name), name)
    if summary.get("method") != protocol["method"] or summary.get("seed") != protocol["seed"]:
        raise ValueError("Run summary method/seed disagree with the protocol")
    equal_number(cost["mission_seconds"], previous_mission, "mission_seconds")
    equal_number(cost["path_length_m"], previous_path, "path_length_m")
    for name in ("planning_seconds", "mapping_seconds", "sensor_seconds"):
        equal_number(cost[name], sum(step[name] for step in steps), name)
    equal_number(cost["mission_seconds"], cost["planning_seconds"] + cost["mapping_seconds"] + cost["simulated_flight_seconds"], "mission cost decomposition")
    equal_number(cost["checkpoint_count"], len(checkpoints), "checkpoint_count")
    equal_number(cost["optimizer_steps"], events * recipe["mapping_optimizer_steps_per_event"], "optimizer_steps")
    expected_calls = 0 if protocol["method"] == "refine_only" else events - recipe["prefix"]
    equal_number(cost["branch_observation_calls"], expected_calls, "branch_observation_calls")
    if cost["blocked_future_observation_calls"] != 0:
        raise ValueError("The controlled run attempted a forbidden future observation")
    if recipe["mode"] == "observations":
        if (summary.get("stop_reason") != "event_limit" or cost.get("stop_reason") != "event_limit"
                or events != recipe["observations"]):
            raise ValueError("Observation protocol did not stop at its requested event limit")
    else:
        budget = number(recipe.get("seconds"), "time budget")
        if budget <= 0 or summary.get("stop_reason") != "time_budget" or cost.get("stop_reason") != "time_budget":
            raise ValueError("Time protocol must finish by its actual time budget")
        if previous_mission < budget or steps[recipe["prefix"] - 1]["mission_seconds"] >= budget:
            raise ValueError("Time protocol budget was not completed after a usable prefix")
        if any(step["mission_seconds"] >= budget for step in steps[:-1]):
            raise ValueError("Time protocol continued after exhausting its budget")
        if events > recipe["safety_event_cap"]:
            raise ValueError("Time protocol exceeded its safety event cap")


def read_run(experiment):
    protocol = json.loads((experiment / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((experiment / "final_result.json").read_text(encoding="utf-8"))
    if protocol.get("version") != VERSION:
        raise ValueError(f"Unsupported/original protocol: {experiment}")
    seed = protocol.get("seed")
    if type(seed) is not int or seed < 0 or protocol.get("method") not in METHODS:
        raise ValueError("Method and RNG seed must be recorded explicitly")
    if not isinstance(protocol.get("scene"), str) or not protocol["scene"]:
        raise ValueError("Scene identity must be recorded explicitly")
    for key in ("scene_mesh_sha256", "context_hash"):
        digest(protocol.get(key), key)
    recipe = protocol["protocol"]
    if recipe.get("mode") not in ("observations", "time"):
        raise ValueError("Unsupported controlled budget mode")
    for key in ("prefix", "observations", "mapping_optimizer_steps_per_event", "safety_event_cap"):
        if type(recipe.get(key)) is not int or recipe[key] <= 0:
            raise ValueError(f"Positive integer protocol field required: {key}")
    if recipe["prefix"] >= recipe["observations"] or recipe["prefix"] >= recipe["safety_event_cap"]:
        raise ValueError("Protocol limits must extend past the common prefix")
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
    if any(type(value) is not int or value <= 0 for field in ("step", "update_event", "observation_count") for value in result[field]):
        raise ValueError("Controlled event and observation counts must be positive integers")
    check_file = experiment / "artifact-check.json"
    if not check_file.is_file():
        raise ValueError("Completed artifact validation is required before aggregation")
    check = json.loads(check_file.read_text(encoding="utf-8"))
    if (check.get("artifact_chain_passed") is not True or check.get("checkpoint_count") != n
            or check.get("observation_count") != result["observation_count"]
            or [item.get("checkpoint") for item in check.get("artifacts", [])] != result["step"]):
        raise ValueError("Artifact validation does not match the completed checkpoints")
    # A saved success report can survive deleted/truncated/replaced artifacts.
    # Reinspect the actual files every time before statistics are produced.
    fresh = check_current_artifacts(experiment)
    if check.get("files") is not None and check["files"] != fresh["files"]:
        raise ValueError("Artifact validation fingerprints are stale; revalidate current files")
    if check.get("last_checkpoint_metrics") is not None and check["last_checkpoint_metrics"] != fresh["last_checkpoint_metrics"]:
        raise ValueError("Artifact validation metrics are stale; revalidate current files")
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
    digest(protocol.get("prefix_sha256"), "prefix_sha256")
    digest(protocol.get("prefix_camera_sha256"), "prefix_camera_sha256")
    audit_execution(experiment, protocol, result)
    return {"method": protocol["method"], "seed": seed, "protocol": protocol,
            "result": result, "cost": cost, "experiment": experiment}


def stats(values):
    return {"mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def aggregate(experiments, methods, expected_seeds):
    if (not methods or len(methods) != len(set(methods)) or any(method not in METHODS for method in methods)
            or not expected_seeds or len(expected_seeds) != len(set(expected_seeds))
            or any(type(seed) is not int or seed < 0 for seed in expected_seeds)):
        raise ValueError("Requested methods/seeds must be nonempty, valid and unique")
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
