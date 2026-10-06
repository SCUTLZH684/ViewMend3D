"""Read-only, complete-chain v2 analysis; development/heldout and budgets stay separate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from viewmend3d.campaign_profiles import get_profile, campaign_spec_sha256, validate_profile_state
from viewmend3d.protocol import Protocol
from viewmend3d.diagnostic_audit import audit_guarded_run
from aggregate_benchmark import aggregate
from check_artifacts import check, ply_counts, read_cameras
from analyze_campaign import (Pending, read, number, stats, METRICS, COSTS,
                              live_registered_handle, analyze_diagnostics)

NAME = "optimization-v2"


def completed_campaign(root, fingerprints):
    path = root / f"runs/campaigns/{NAME}/state.json"
    if not path.is_file():
        raise Pending("campaign_not_established")
    state = read(path, fingerprints)
    profile = validate_profile_state(state, NAME)
    stages = profile["stages"]
    if state.get("status") != "completed" or state.get("stage_index") != len(stages):
        raise Pending("campaign_not_complete", {"campaign_status": state.get("status"),
                      "stage_index": state.get("stage_index"), "updated_at": state.get("updated_at")})
    attempts = state.get("attempts")
    if not isinstance(attempts, list) or not isinstance(state.get("source_identity"), dict):
        raise ValueError("A frozen source and complete attempt ledger are required")
    active = state.get("active")
    if active and (active.get("child_launching") or live_registered_handle(active)):
        raise Pending("campaign_process_still_live")
    completed = {}
    directory = (root / f"runs/campaigns/{NAME}/stages").resolve()
    for attempt in attempts:
        if attempt.get("status") != "completed":
            continue
        index = attempt.get("stage_index")
        if type(index) is not int or not 0 <= index < len(stages):
            raise ValueError("Unknown completed v2 stage")
        stage = stages[index]
        name = stage["name"]
        count = len(stage["methods"]) * len(stage["seeds"])
        if name in completed or attempt.get("stage") != name or attempt.get("validated_experiments") != count:
            raise ValueError("A v2 stage is duplicated or lacks its exact validation count")
        run = Path(attempt["run_dir"]).resolve()
        if run.parent != directory or not re.fullmatch(re.escape(name) + r"-[0-9a-f]{32}", run.name):
            raise ValueError("A v2 stage is outside its managed directory")
        if attempt.get("child_launching") or live_registered_handle(attempt):
            raise Pending("completed_stage_process_still_live", {"stage": name})
        completed[name] = run
    if set(completed) != {stage["name"] for stage in stages}:
        raise ValueError("All five independently validated v2 stages are required")
    return state, profile, completed


def stage_gate(run, stage, fingerprints):
    summary = read(run / "benchmark-summary.json", fingerprints)
    if summary.get("status") != "completed":
        raise Pending("benchmark_stage_not_complete", {"stage": stage["name"]})
    expected_protocol = Protocol(observations=stage["frames"], prefix=stage["prefix"],
                                mode=stage["mode"], seconds=stage["seconds"], recipe=NAME,
                                campaign_spec_sha256=campaign_spec_sha256(NAME)).validate().as_dict()
    expected = {(method, seed) for method in stage["methods"] for seed in stage["seeds"]}
    entries = summary.get("experiments", [])
    actual = [(entry.get("method"), entry.get("seed")) for entry in entries]
    if (summary.get("scene") != "replica/office0" or summary.get("methods") != stage["methods"]
            or summary.get("seeds") != stage["seeds"] or len(actual) != len(expected)
            or set(actual) != expected or any(entry.get("status") != "completed" for entry in entries)
            or summary.get("protocol") != expected_protocol or summary.get("future_observation_calls") != 0
            or any(summary.get(key) != expected_protocol[key]
                   for key in ("recipe", "recipe_sha256", "campaign_spec_sha256"))):
        raise ValueError("V2 stage methods/seeds/budget/recipe differ from the complete preregistered suite")
    if not (run / "paired-results.json").is_file():
        raise Pending("paired_report_not_ready", {"stage": stage["name"]})
    return summary


def paired_statistics(runs, method, baseline, seeds, families):
    selected = {run["seed"]: run for run in runs if run["method"] == method}
    reference = {run["seed"]: run for run in runs if run["method"] == baseline}
    if set(selected) != set(seeds) or set(reference) != set(seeds):
        raise ValueError("The complete declared seed pair is required")
    result = {}
    for family, keys in families.items():
        result[family] = {}
        for key in keys:
            differences = [selected[seed][family][key] - reference[seed][family][key] for seed in seeds]
            result[family][key] = {**stats(differences),
                "method_minus_baseline_by_seed": dict(zip(map(str, seeds), differences)),
                "relative_percent_by_seed": {str(seed): 100 * differences[index] / reference[seed][family][key]
                                             if reference[seed][family][key] else None for index, seed in enumerate(seeds)},
                "negative_difference_seeds": sum(value < 0 for value in differences),
                "positive_difference_seeds": sum(value > 0 for value in differences)}
    return result


def audit_export(root, run, stage, experiment, protocol, result, fingerprints):
    token = run.name.rsplit("-", 1)[1]
    folder = root / "runs/web-assets" / f"campaign-{NAME}-{stage['name']}-{token}-{protocol['method']}-s{protocol['seed']}"
    manifest = read(folder / "manifest.json", fingerprints)
    if (manifest.get("status") != "completed" or manifest.get("method_id") != protocol["method"]
            or manifest.get("seed") != protocol["seed"] or manifest.get("protocol") != protocol
            or len(manifest.get("checkpoints", [])) != len(result["step"])):
        raise ValueError("A v2 export is missing or has a different method/seed/protocol")
    for index, checkpoint in enumerate(manifest["checkpoints"]):
        event = result["step"][index]
        if (checkpoint.get("step") != event or checkpoint.get("update_event") != event
                or checkpoint.get("observation_count") != result["observation_count"][index]
                or checkpoint.get("camera_count") != result["observation_count"][index]):
            raise ValueError("Export acquisition/update counts disagree with the actual result")
        for name, (source, factor) in METRICS.items():
            if not math.isclose(number(checkpoint.get(name)), result[source][index] * factor, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Exported metric is not the full-mesh evaluation result")
        preview = folder / checkpoint["mesh"]
        if preview.parent.resolve() != folder.resolve():
            raise ValueError("Exported preview path leaves its managed directory")
        counts = ply_counts(preview)
        if {"vertices": counts["vertex"], "faces": counts["face"]} != checkpoint.get("preview"):
            raise ValueError("Exported preview vertex/face counts disagree with its actual PLY")
        fingerprints[str(preview)] = hashlib.sha256(preview.read_bytes()).hexdigest()
    cameras = read_cameras(experiment / "map" / f"cameras_{result['step'][-1]:03d}.pkl")
    if (manifest.get("camera_convention") != "OpenCV camera-to-world"
            or len(manifest.get("cameras", [])) != len(cameras) or len(manifest.get("path_stops", [])) != len(cameras)):
        raise ValueError("Export camera and trajectory endpoint counts are incomplete")
    trajectory = manifest.get("trajectory", [])
    for row, camera, stop in zip(cameras, manifest["cameras"], manifest["path_stops"]):
        position = row[3:12:4]
        expected = {"position": position, "rotation": [row[r * 4 + c] for r in range(3) for c in range(3)],
                    "intrinsic": row[16:]}
        for field, values in expected.items():
            actual = camera.get(field) if isinstance(camera, dict) else None
            if (not isinstance(actual, list) or len(actual) != len(values)
                    or any(abs(number(value) - reference) > 1e-5 for value, reference in zip(actual, values))):
                raise ValueError("Export camera position/rotation/intrinsic differs from actual acquired calibration")
        if (type(stop) is not int or not 1 <= stop <= len(trajectory)
                or math.dist(camera["position"], position) > 1e-5
                or math.dist(trajectory[stop - 1], position) > 1e-4):
            raise ValueError("Export camera/trajectory endpoint differs from the actual acquired pose")
    return folder.name


def analyze_stage(root, run, stage, summary, fingerprints):
    methods, seeds = stage["methods"], stage["seeds"]
    experiments = [run / f"experiments/benchmark/replica/office0/{method}/{seed}" for method in methods for seed in seeds]
    fresh = aggregate(experiments, methods, seeds)
    if fresh != read(run / "paired-results.json", fingerprints):
        raise ValueError("Saved v2 paired results disagree with freshly audited current artifacts/diagnostics")
    runs, diagnostics, exports = [], [], []
    for experiment in experiments:
        protocol = read(experiment / "protocol.json", fingerprints)
        result = read(experiment / "final_result.json", fingerprints)
        cost = read(experiment / "run-summary.json", fingerprints)
        ledger = read(experiment / "steps.json", fingerprints)
        read(experiment / "exp_config.json", fingerprints)
        for path in (experiment / "diagnostics").glob("step_*.json"):
            read(path, fingerprints)
        actual_files = check(experiment)["files"]
        for name, identity in actual_files.items():
            fingerprints[str(experiment / name)] = identity["sha256"]
        branch = {key: math.fsum(number(entry[key]) for entry in ledger if entry["event"] > stage["prefix"])
                  for key in ("planning_seconds", "mapping_seconds", "sensor_seconds", "diagnostic_seconds")}
        branch.update(events=cost["events"] - stage["prefix"], observations=cost["branch_observation_calls"])
        runs.append({"method": protocol["method"], "seed": protocol["seed"],
                     "metrics": {name: number(result[source][-1]) * factor for name, (source, factor) in METRICS.items()},
                     "cost": {key: number((result["cost"] if key == "meshing_evaluation_wall_seconds" else cost)[key]) for key in COSTS},
                     "post_prefix_cost": branch, "prefix_sha256": protocol["prefix_sha256"],
                     "prefix_camera_sha256": protocol["prefix_camera_sha256"],
                     "wall_time_scope": cost["wall_time_scope"], "memory_scope": cost["memory_scope"]})
        guarded = audit_guarded_run(experiment, protocol)
        diagnostics.append({"method": protocol["method"], "seed": protocol["seed"], **guarded} if guarded is not None else
                           analyze_diagnostics(experiment, protocol["method"], protocol["seed"], stage["prefix"], cost["events"], fingerprints))
        exports.append(audit_export(root, run, stage, experiment, protocol, result, fingerprints))
    families = {"metrics": tuple(METRICS), "cost": COSTS, "post_prefix_cost": tuple(runs[0]["post_prefix_cost"])}
    by_method = {method: {family: {key: stats([row[family][key] for row in runs if row["method"] == method]) for key in keys}
                         for family, keys in families.items()} for method in methods}
    return {"stage": stage["name"], "partition": "smoke" if stage["name"] == "smoke" else stage["name"].split("_")[0],
            "run_id": run.name, "seeds": seeds, "status": "completed_and_freshly_audited",
            **{key: fresh[key] for key in ("protocol", "source_versions", "evaluation", "scene", "scene_mesh_sha256",
                                          "scene_assets_sha256", "context_hash", "recipe", "recipe_sha256", "campaign_spec_sha256")},
            "pipeline_wall_seconds": summary["pipeline_wall_seconds"], "method_statistics": by_method, "per_run": runs,
            "paired_difference_vs_confidence": {method: paired_statistics(runs, method, "confidence_nooracle", seeds, families)
                                                for method in methods if method != "confidence_nooracle"},
            "diagnostics_per_run": diagnostics, "verified_exports": exports,
            "fresh_artifact_audit": True, "saved_paired_report_verified_against_current_files": True}


def analyze(root):
    root = Path(root).resolve()
    fingerprints = {}
    state, profile, paths = completed_campaign(root, fingerprints)
    stages = {}
    for stage in profile["stages"]:
        run = paths[stage["name"]]
        summary = stage_gate(run, stage, fingerprints)
        stages[stage["name"]] = analyze_stage(root, run, stage, summary, fingerprints)
    identities = [{key: item[key] for key in ("source_versions", "scene_mesh_sha256", "scene_assets_sha256", "evaluation",
                                             "recipe_sha256", "campaign_spec_sha256")} for item in stages.values()]
    if any(item != identities[0] for item in identities):
        raise ValueError("V2 stages changed their frozen source/assets/recipe/evaluator")
    for name, expected in fingerprints.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("A v2 input changed during the fresh complete-chain audit")
    return {"version": "viewmend-campaign-analysis-v2", "status": "complete",
            "generated_at": datetime.now(timezone.utc).isoformat(), "scene": "replica/office0", "campaign": NAME,
            "campaign_spec_sha256": campaign_spec_sha256(NAME), "campaign_project_source_identity": state["source_identity"],
            "completed_campaign_stages": 5, "validated_campaign_experiments": 27, "quality_runs": 24,
            "development_seeds": [0, 1], "heldout_seeds": [3, 4, 5], "primary_method": "defect_guarded",
            "scope": "Four separate single-scene paired comparisons: development n=2, heldout seeds n=3; separate budgets. Smoke excluded from quality conclusions.",
            "units": {"accuracy_cm": "cm", "completion_cm": "cm", "coverage_percent": "% / paired pp", "chamfer_mm": "mm"},
            "limitations": ["Seed holdout is not scene holdout; office0 only.", "Descriptive sample standard deviations are not confidence intervals.",
                            "A bounded proxy-score regret is not a ground-truth quality guarantee.",
                            "Utility time includes rendering/visibility/scoring; Torch peak excludes Habitat/OpenGL.",
                            "Fewer observations with lower time/memory do not prove equal-quality acceleration.",
                            "Current custom CUDA kernels may retain numerical nondeterminism."],
            "stages": stages, "source_input_sha256": fingerprints, "process_observation_platform": sys.platform}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/ViewMend3D"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = analyze(args.root)
        if args.output:
            target, root = args.output.resolve(), args.root.resolve()
            if target.is_relative_to(root) and not (target.is_relative_to(root / "setup") or target.is_relative_to(root / "runs")):
                raise ValueError("Analysis output must be new scratch/setup/runs, outside frozen source")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            print(json.dumps({"status": "complete", "output": str(target), "audited_runs": 27}))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Pending as error:
        print(json.dumps({"status": "pending", "quality_summary_generated": False,
                          "reason": error.reason, **error.details}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
