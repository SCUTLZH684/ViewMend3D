"""Read-only final analysis of a verified optimization-v1 campaign (stdlib).

Run only against our trusted server outputs. Until the complete campaign is
validated this prints a pending record, exits 2, and writes no report. Frozen
repository files and experiment outputs are never modified. The existing
stdlib artifact/aggregation auditor is imported only after completion gates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

MAIN = ("confidence_nooracle", "random_matched", "defect")
STAGES = {"smoke": (0, 3), "observations": (1, 15), "time": (2, 9)}
METHODS = {"observations": MAIN + ("defect_no_gate", "refine_only"), "time": MAIN}
SEEDS = (0, 1, 2)
METRICS = {"accuracy_cm": ("mesh_accuracy", 1), "completion_cm": ("mesh_completion", 1),
           "coverage_percent": ("mesh_completion_ratio", 1), "chamfer_mm": ("mesh_chamfer_distance", 1000)}
COSTS = ("mission_seconds", "wall_seconds", "planning_seconds", "mapping_seconds", "sensor_seconds",
         "diagnostic_seconds", "simulated_flight_seconds", "path_length_m", "observations", "events",
         "optimizer_steps", "additional_optimizer_steps", "peak_torch_allocated_mb",
         "meshing_evaluation_wall_seconds")
SCORE_TOLERANCE = 1e-6


class Pending(Exception):
    def __init__(self, reason, details=None):
        self.reason, self.details = reason, details or {}


def read(path, fingerprints=None):
    raw = Path(path).read_bytes()
    def reject_constant(value):
        raise ValueError("Non-finite JSON constant")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique)
    if fingerprints is not None:
        fingerprints[str(path)] = hashlib.sha256(raw).hexdigest()
    return value


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite numeric value required")
    return value


def checked_scene_assets_sha256(protocol, expected=None):
    """Require the complete scene-asset digest in every branch protocol."""
    digest = protocol.get("scene_assets_sha256")
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Branch protocol requires a valid scene_assets_sha256")
    if expected is not None and digest != expected:
        raise ValueError("Branch scene_assets_sha256 differs within the stage")
    return digest


def verify_stage_source_identity(results):
    observations, time = results["observations"], results["time"]
    for result in (observations, time):
        checked_scene_assets_sha256(result)
    if any(observations[key] != time[key] for key in
           ("source_versions", "scene_mesh_sha256", "scene_assets_sha256")):
        raise ValueError("Campaign stages changed source versions or scene assets")


def stats(values):
    values = [number(value) for value in values]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def distribution(values):
    values = [number(value) for value in values]
    return {**stats(values), "minimum": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "maximum": max(values) if values else None}


def normalize(values):
    values = [max(0.0, number(value)) for value in values]
    total = math.fsum(values)
    return [value / total if total else 0.0 for value in values]


def ranking(scores):
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def live_registered_handle(active):
    """A starttime match remains a live handle even during an exec argv window."""
    if sys.platform != "linux" or not active:
        return False
    for kind in ("worker", "child"):
        for key in (kind, f"{kind}_start"):
            identity = active.get(key)
            if not identity:
                continue
            try:
                process = Path("/proc") / str(identity["pid"])
                raw = (process / "stat").read_text()
                fields = raw[raw.rfind(")") + 2:].split()
                if fields[0] != "Z" and fields[19] == identity["starttime"]:
                    return True
            except FileNotFoundError:
                continue
            except (OSError, IndexError, KeyError, TypeError):
                raise Pending("registered_process_observation_unavailable")
    return False


def completed_campaign(root, fingerprints):
    path = root / "runs/campaigns/optimization-v1/state.json"
    if not path.exists():
        raise Pending("campaign_not_established")
    state = read(path, fingerprints)
    if state.get("version") != "viewmend-campaign-v1":
        raise ValueError("Unsupported campaign schema")
    if state.get("status") != "completed" or state.get("stage_index") != 3:
        raise Pending("campaign_not_complete", {"campaign_status": state.get("status"),
                      "stage_index": state.get("stage_index"), "updated_at": state.get("updated_at")})
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("Campaign attempts missing")
    completed = {}
    directory = (root / "runs/campaigns/optimization-v1/stages").resolve()
    for attempt in attempts:
        if attempt.get("status") != "completed":
            continue
        stage = attempt.get("stage")
        if stage not in STAGES or stage in completed:
            raise ValueError("Unknown or duplicate completed stage")
        index, count = STAGES[stage]
        if attempt.get("stage_index") != index or attempt.get("validated_experiments") != count:
            raise ValueError("Completed stage lacks its exact validation count")
        run = Path(attempt["run_dir"]).resolve()
        if run.parent != directory or not re.fullmatch(stage + r"-[0-9a-f]{32}", run.name):
            raise ValueError("Run is outside the managed campaign stage directory")
        if live_registered_handle(attempt) or attempt.get("child_launching") is True:
            raise Pending("completed_stage_process_still_live", {"stage": stage})
        completed[stage] = run
    if set(completed) != set(STAGES):
        raise ValueError("All three validated stages are required")
    return state, completed


def stage_gate(run, stage, fingerprints):
    summary = read(run / "benchmark-summary.json", fingerprints)
    methods = METHODS[stage]
    expected = {(method, seed) for method in methods for seed in SEEDS}
    entries = summary.get("experiments", [])
    actual = [(entry.get("method"), entry.get("seed")) for entry in entries]
    if summary.get("status") != "completed":
        raise Pending("benchmark_stage_not_complete", {"stage": stage, "benchmark_status": summary.get("status")})
    if (summary.get("scene") != "replica/office0" or summary.get("methods") != list(methods)
            or summary.get("seeds") != list(SEEDS) or len(actual) != len(expected)
            or set(actual) != expected or any(entry.get("status") != "completed" for entry in entries)):
        raise ValueError("Completed benchmark is not the full requested method/seed suite")
    recipe = summary.get("protocol", {})
    for key, expected_value in {"mode": stage, "observations": 60, "prefix": 20,
            "seconds": 180.0, "candidate_count": 100, "roi_count": 30,
            "sample_points": 500000, "future_candidate_depth_mask": False,
            "mapping_optimizer_steps_per_event": 10}.items():
        if recipe.get(key) != expected_value:
            raise ValueError(f"Frozen stage differs: {stage}/{key}")
    if summary.get("future_observation_calls") != 0:
        raise ValueError("Benchmark attempted forbidden future observations")
    if not (run / "paired-results.json").is_file():
        raise Pending("paired_report_not_ready", {"stage": stage})
    return summary


def analyzer_module(root):
    # These two repository modules use only Python's standard library and
    # inspect current trusted artifacts without loading Torch tensors.
    sys.path.insert(0, str(root / "scripts/activegs"))
    path = root / "scripts/activegs/aggregate_benchmark.py"
    spec = importlib.util.spec_from_file_location("campaign_result_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_statistics(runs, method, baseline, fields):
    selected = {run["seed"]: run for run in runs if run["method"] == method}
    reference = {run["seed"]: run for run in runs if run["method"] == baseline}
    if set(selected) != set(SEEDS) or set(reference) != set(SEEDS):
        raise ValueError("A complete three-seed pair is required")
    result = {}
    for family, keys in fields.items():
        result[family] = {}
        for key in keys:
            differences = [selected[seed][family][key] - reference[seed][family][key] for seed in SEEDS]
            relative = {str(seed): (100 * differences[index] / reference[seed][family][key]
                        if reference[seed][family][key] != 0 else None) for index, seed in enumerate(SEEDS)}
            result[family][key] = {**stats(differences), "method_minus_baseline_by_seed": dict(zip(map(str, SEEDS), differences)),
                "relative_percent_by_seed": relative, "negative_difference_seeds": sum(value < 0 for value in differences),
                "positive_difference_seeds": sum(value > 0 for value in differences)}
    return result


def analyze_diagnostics(experiment, method, seed, prefix, events, fingerprints):
    expected = [] if method == "refine_only" else list(range(prefix + 1, events + 1))
    files = sorted((experiment / "diagnostics").glob("step_*.json"))
    selected_files = {}
    for path in files:
        match = re.fullmatch(r"step_(\d+)\.json", path.name)
        if not match:
            raise ValueError("Unexpected diagnostic filename")
        event = int(match[1])
        if event > prefix:
            selected_files[event] = path
    if set(selected_files) != set(expected):
        raise ValueError("Complete post-prefix diagnostic ledger required")
    rows, valid, positive = [], [], []
    for event in expected:
        data = read(selected_files[event], fingerprints)
        if (data.get("step") != event or data.get("method") != method or data.get("status") != "selected"
                or data.get("future_candidate_observation_queries") != 0
                or data.get("planning_mask_source") != "current_map"):
            raise ValueError("Diagnostic identity/information boundary mismatch")
        row = {"event": event, "geometry_measured": data.get("geometry_measured") is True,
               "geometry_active": data.get("geometry_active") is True,
               "all_zero_utility_fallback": data.get("all_zero_utility_fallback") is True,
               "utility_seconds": number(data["utility_seconds"]),
               "sampling_seconds": number(data["sampling_seconds"]),
               "path_scoring_seconds": number(data["path_scoring_seconds"]),
               "planner_internal_seconds": number(data["planning_seconds"]),
               "diagnostics_write_seconds_excluding_final_json": number(data["diagnostics_seconds"])}
        if row["geometry_measured"]:
            reachable = data["reachable"]
            count = len(data["base"])
            if (count != 100 or len(reachable) != count or len(data["defect"]) != count
                    or len(data["valid_fraction"]) != count or len(data["positive_fraction"]) != count
                    or data.get("normalization_domain") != "reachable_candidates"):
                raise ValueError("Misaligned geometric candidate arrays")
            base = [number(value) if reachable[index] else 0 for index, value in enumerate(data["base"])]
            defect = [number(value) if reachable[index] else 0 for index, value in enumerate(data["defect"])]
            lengths = [number(value) if reachable[index] else 0 for index, value in enumerate(data["path_lengths"])]
            if not any(reachable) or any(value < 0 for value in base + defect + lengths):
                raise ValueError("Invalid nonnegative reachable utilities/path lengths")
            nbase, ndefect, costs = normalize(base), normalize(defect), normalize(lengths)
            threshold, weight = data["geometry_thresholds"]["minimum_peak"], data["geometry_weight"]
            active = max(defect) >= threshold and math.fsum(defect) > 0 and weight > 0
            if active != row["geometry_active"]:
                raise ValueError("Recorded geometry fallback disagrees with its actual reachable signal")
            utility = normalize([value + weight * ndefect[index] * active for index, value in enumerate(nbase)])
            scores = [value - data["path_length_factor"] * costs[index] if reachable[index] else -math.inf for index, value in enumerate(utility)]
            recorded = [number(value) if reachable[index] else -math.inf for index, value in enumerate(data["final_scores"])]
            selected = data["selected_index"]
            if ranking(recorded)[0] != selected:
                raise ValueError("Selected candidate is not the first maximum recorded score")
            error = max(abs(scores[index] - recorded[index]) for index in range(count) if reachable[index])
            if not row["all_zero_utility_fallback"] and error > SCORE_TOLERANCE:
                raise ValueError("Recomputed real-valued score disagrees with recorded float32 ranking")
            valid.extend(number(value) for value in data["valid_fraction"])
            positive.extend(number(value) for value in data["positive_fraction"])
            base_scores = [value - data["path_length_factor"] * costs[index] if reachable[index] else -math.inf for index, value in enumerate(nbase)]
            ordered = ranking(base_scores)
            available = math.fsum(base) > 0
            gap = base_scores[ordered[0]] - base_scores[ordered[1]] if sum(reachable) > 1 else math.inf
            robust = available and gap > 2 * SCORE_TOLERANCE
            row.update(defect_max_reachable=max(defect), weak_or_zero_geometry_fallback=not active,
                selected_valid_fraction=data["valid_fraction"][selected], selected_positive_fraction=data["positive_fraction"][selected],
                lambda_zero_available=available, lambda_zero_near_tie=available and not robust,
                lambda_zero_index=ordered[0] if available else None,
                lambda_zero_selection_changed=ordered[0] != selected if available else None,
                lambda_zero_robust=robust, score_formula_max_abs_error=error if not row["all_zero_utility_fallback"] else None)
        rows.append(row)
    measured = [row for row in rows if row["geometry_measured"]]
    counterfactual = [row for row in measured if row["lambda_zero_available"]]
    robust = [row for row in measured if row["lambda_zero_robust"]]
    total = len(rows)
    return {"method": method, "seed": seed, "post_prefix_events": total,
        "geometry_measured_events": len(measured), "geometry_active_events": sum(row["geometry_active"] for row in measured),
        "geometry_active_rate": sum(row["geometry_active"] for row in measured) / len(measured) if measured else None,
        "weak_or_zero_geometry_fallback_events": sum(row["weak_or_zero_geometry_fallback"] for row in measured),
        "all_zero_utility_fallback_events": sum(row["all_zero_utility_fallback"] for row in rows),
        "lambda_zero_available_events": len(counterfactual),
        "lambda_zero_selection_changed_events": sum(row["lambda_zero_selection_changed"] for row in counterfactual),
        "lambda_zero_near_tie_events": len(counterfactual) - len(robust),
        "lambda_zero_robust_events": len(robust),
        "lambda_zero_robust_changed_events": sum(row["lambda_zero_selection_changed"] for row in robust),
        "all_candidate_valid_fraction": distribution(valid), "all_candidate_positive_fraction": distribution(positive),
        "selected_valid_fraction": distribution([row["selected_valid_fraction"] for row in measured]),
        "utility_seconds": distribution([row["utility_seconds"] for row in rows]),
        "utility_seconds_total": math.fsum(row["utility_seconds"] for row in rows),
        "sampling_seconds_total": math.fsum(row["sampling_seconds"] for row in rows),
        "path_scoring_seconds_total": math.fsum(row["path_scoring_seconds"] for row in rows),
        "diagnostics_json_seconds_scope": "Persisted per-event field excludes its own final JSON write; use run-summary diagnostic_seconds for complete timed IO.",
        "events": rows}


def analyze_stage(root, run, stage, summary, auditor, fingerprints):
    methods = METHODS[stage]
    experiments = [run / f"experiments/benchmark/replica/office0/{method}/{seed}" for method in methods for seed in SEEDS]
    stored = read(run / "paired-results.json", fingerprints)
    fresh = auditor.aggregate(experiments, list(methods), list(SEEDS))
    if fresh != stored:
        raise ValueError("Saved paired report does not exactly match freshly audited current files")
    runs, diagnostic_runs = [], []
    scene_assets_sha256 = None
    for method in methods:
        for seed in SEEDS:
            experiment = run / f"experiments/benchmark/replica/office0/{method}/{seed}"
            protocol = read(experiment / "protocol.json", fingerprints)
            scene_assets_sha256 = checked_scene_assets_sha256(protocol, scene_assets_sha256)
            result = read(experiment / "final_result.json", fingerprints)
            cost = read(experiment / "run-summary.json", fingerprints)
            ledger = read(experiment / "steps.json", fingerprints)
            # The frozen artifact aggregator has already audited protocol,
            # stopping reason, units, event ledger and current binary files.
            metric_values = {name: number(result[source][-1]) * factor for name, (source, factor) in METRICS.items()}
            cost_values = {key: number((result["cost"] if key == "meshing_evaluation_wall_seconds" else cost)[key]) for key in COSTS}
            post_prefix = [entry for entry in ledger if entry["event"] > protocol["protocol"]["prefix"]]
            branch = {key: math.fsum(number(entry[key]) for entry in post_prefix)
                      for key in ("planning_seconds", "mapping_seconds", "sensor_seconds", "diagnostic_seconds")}
            branch.update(events=len(post_prefix), observations=cost["branch_observation_calls"])
            runs.append({"method": method, "seed": seed, "metrics": metric_values, "cost": cost_values,
                         "post_prefix_cost": branch, "prefix_sha256": protocol["prefix_sha256"],
                         "prefix_camera_sha256": protocol["prefix_camera_sha256"],
                         "wall_time_scope": cost["wall_time_scope"], "memory_scope": cost["memory_scope"]})
            diagnostic_runs.append(analyze_diagnostics(experiment, method, seed, protocol["protocol"]["prefix"], cost["events"], fingerprints))
    families = {"metrics": tuple(METRICS), "cost": COSTS, "post_prefix_cost": tuple(runs[0]["post_prefix_cost"])}
    by_method = {method: {family: {key: stats([entry[family][key] for entry in runs if entry["method"] == method]) for key in keys}
                  for family, keys in families.items()} for method in methods}
    signals = {}
    for method in methods:
        entries = [entry for entry in diagnostic_runs if entry["method"] == method]
        measured = sum(entry["geometry_measured_events"] for entry in entries)
        active = sum(entry["geometry_active_events"] for entry in entries)
        signals[method] = {"post_prefix_events": sum(entry["post_prefix_events"] for entry in entries),
            "geometry_measured_events": measured, "geometry_active_events": active,
            "pooled_geometry_active_rate": active / measured if measured else None,
            "per_seed_geometry_active_rate": stats([entry["geometry_active_rate"] for entry in entries if entry["geometry_active_rate"] is not None]),
            "weak_or_zero_geometry_fallback_events": sum(entry["weak_or_zero_geometry_fallback_events"] for entry in entries),
            "all_zero_utility_fallback_events": sum(entry["all_zero_utility_fallback_events"] for entry in entries),
            "lambda_zero_robust_events": sum(entry["lambda_zero_robust_events"] for entry in entries),
            "lambda_zero_robust_changed_events": sum(entry["lambda_zero_robust_changed_events"] for entry in entries),
            "utility_seconds_total_by_seed": {str(entry["seed"]): entry["utility_seconds_total"] for entry in entries}}
    return {"stage": stage, "run_id": run.name, "status": "completed_and_freshly_audited",
        "protocol": fresh["protocol"], "source_versions": fresh["source_versions"], "evaluation": fresh["evaluation"],
        "scene": fresh["scene"], "scene_mesh_sha256": fresh["scene_mesh_sha256"],
        "scene_assets_sha256": scene_assets_sha256, "context_hash": fresh["context_hash"],
        "pipeline_wall_seconds": summary["pipeline_wall_seconds"],
        "method_statistics": by_method, "per_run": runs,
        "paired_difference_vs_confidence": {method: paired_statistics(runs, method, "confidence_nooracle", families) for method in methods if method != "confidence_nooracle"},
        "ablation_difference_vs_defect": {method: paired_statistics(runs, method, "defect", families) for method in ("defect_no_gate", "refine_only") if method in methods},
        "diagnostic_statistics": signals, "diagnostics_per_run": diagnostic_runs,
        "fresh_artifact_audit": True, "saved_paired_report_verified_against_current_files": True}


def analyze(root):
    root = Path(root).resolve()
    fingerprints = {}
    state, stages = completed_campaign(root, fingerprints)
    summaries = {stage: stage_gate(stages[stage], stage, fingerprints) for stage in METHODS}
    auditor = analyzer_module(root)
    results = {stage: analyze_stage(root, stages[stage], stage, summaries[stage], auditor, fingerprints) for stage in METHODS}
    verify_stage_source_identity(results)
    for filename, recorded in fingerprints.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != recorded:
            raise ValueError("Analysis source JSON changed while it was being read")
    return {"version": "viewmend-campaign-analysis-v1", "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(), "scene": "replica/office0", "seeds": list(SEEDS),
        "campaign_project_source_identity": state.get("source_identity"),
        "completed_campaign_stages": 3, "validated_campaign_experiments": 27,
        "scope": "Single-scene, three-seed paired descriptive statistics; observations and time are separate comparisons.",
        "units": {"accuracy_cm": "cm", "completion_cm": "cm", "coverage_percent": "% (differences: percentage points)",
                  "chamfer_mm": "mm", "path_length_m": "m", "peak_torch_allocated_mb": "MiB, PyTorch allocator only"},
        "paired_difference_definition": "method minus baseline; negative distance/cost differences mean lower values; positive coverage difference means higher coverage; observations/events have no intrinsic improvement direction",
        "lambda_zero_scope": "Same recorded candidate group only; pure Python real-valued formula from float32 values, tolerance 1e-6; near ties excluded from robust change counts; no randomized fallback replay.",
        "diagnostic_scope": "Post-prefix planning events only; refinement has no post-prefix planner. Valid/positive fractions are pixel fractions, not true scene errors. Utility timings include renderer, visibility and scoring, not an isolated geometry kernel.",
        "limitations": ["No statistical significance or cross-scene generalization claim.",
                        "Prefix costs are included per method and also separated; cache loading/serialization and startup are outside reconstruction wall time.",
                        "Meshing/evaluation and export are outside mission/reconstruction wall time.",
                        "PyTorch peak memory excludes Habitat/OpenGL and is not whole-GPU peak usage.",
                        "Within-group lambda-zero ranking is a scoring counterfactual, not a formal reconstruction ablation.",
                        "Actual CUDA numerical nondeterminism remains possible."],
        "stages": results, "source_json_sha256": fingerprints,
        "process_observation_platform": sys.platform}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/ViewMend3D"))
    parser.add_argument("--output", type=Path, help="Optional new scratch JSON path; never overwrite an existing report")
    args = parser.parse_args(argv)
    try:
        report = analyze(args.root)
        if args.output:
            target = args.output.resolve()
            frozen = args.root.resolve()
            if target.is_relative_to(frozen) and (not target.is_relative_to(frozen / "setup") and not target.is_relative_to(frozen / "runs")):
                raise ValueError("Output must be scratch/setup/runs, outside frozen repository files")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
            print(json.dumps({"status": "complete", "output": str(target), "stages": list(report["stages"]), "audited_runs": 24}))
        else:
            print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except Pending as pending:
        print(json.dumps({"status": "pending", "reason": pending.reason, **pending.details,
                          "quality_summary_generated": False, "output_written": False}, allow_nan=False))
        return 2
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        print(json.dumps({"status": "rejected", "error": f"{type(error).__name__}: {error}",
                          "quality_summary_generated": False, "output_written": False}, allow_nan=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
