"""Fresh stdlib audit of v2 scores; no renderer, Torch, simulator or GT reads."""
import json
import math
from pathlib import Path

from .protocol import GUARDED_METHODS, optimization_recipe

TOLERANCE = 1e-6


def finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Finite diagnostic value required: {label}")
    return value


def close(actual, expected, label):
    if abs(finite(actual, label) - expected) > TOLERANCE:
        raise ValueError(f"V2 diagnostic formula disagrees: {label}")


def normalize(values):
    total = math.fsum(values)
    return [value / total if total else 0.0 for value in values]


def audit_guarded_event(data, method, event):
    spec = optimization_recipe("optimization-v2")
    count = spec["candidate_count"]
    if (data.get("method") != method or data.get("step") != event
            or data.get("scoring_version") != "bounded_geometry_v2"
            or data.get("geometry_backend") != "batched"
            or data.get("status") != "selected" or data.get("future_candidate_observation_queries") != 0
            or data.get("planning_mask_source") != "current_map"
            or data.get("normalization_domain") != "reachable_candidates"
            or data.get("geometry_normalization") != "peak_reachable_candidates"
            or data.get("geometry_measured") is not True
            or data.get("geometry_measured_count") != count or data.get("geometry_batch_count") != 1
            or data.get("geometry_thresholds") != spec["geometry_thresholds"]
            or data.get("use_depth_gate") is not spec["methods"][method]["use_depth_gate"]
            or data.get("depth_discontinuity_gate") is not spec["methods"][method]["use_depth_gate"]
            or data.get("path_length_factor") != spec["path_length_factor"]
            or data.get("render_ratio") != spec["render_ratio"]
            or data.get("explore_weight") != spec["explore_weight"]
            or data.get("render_resolution") != spec["render_resolution"]
            or data.get("shared_scene_bbox") is not True or data.get("geometry_weight") != 0):
        raise ValueError("V2 diagnostic method, geometry recipe or batch identity is missing/changed")
    vectors = {}
    for key in ("base", "defect", "path_lengths", "reachable", "baseline_scores",
                "geometry_bonus", "defect_peak_normalized", "final_scores"):
        values = data.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"Complete v2 candidate vector required: {key}")
        vectors[key] = values
    reachable = [index for index, value in enumerate(vectors["path_lengths"])
                 if value is not None and type(value) in (int, float) and math.isfinite(value) and value >= 0]
    if (not reachable or any(type(value) is not bool for value in vectors["reachable"])
            or vectors["reachable"] != [index in reachable for index in range(count)]):
        raise ValueError("V2 reachable domain is invalid")
    beta = spec["methods"][method]["geometry_beta"]
    cap = beta / len(reachable)
    close(data.get("geometry_beta"), beta, "beta")
    close(data.get("geometry_bonus_cap"), cap, "bonus cap")
    if data.get("n_reachable") != len(reachable):
        raise ValueError("V2 reachable candidate count disagrees")
    base = [max(0.0, finite(vectors["base"][index], "base")) for index in reachable]
    base_zero = not sum(base) > 0
    if data.get("all_zero_utility_fallback") is not base_zero:
        raise ValueError("V2 all-zero utility fallback flag disagrees with actual reachable base")
    defect = [max(0.0, finite(vectors["defect"][index], "defect")) for index in reachable]
    lengths = [finite(vectors["path_lengths"][index], "path length") for index in reachable]
    peak, spread = max(defect), max(defect) - min(defect)
    signal = peak >= spec["geometry_thresholds"]["minimum_peak"] and peak > 0
    discriminative = spread > 0
    active = beta > 0 and sum(base) > 0 and len(reachable) > 1 and signal and discriminative
    close(data.get("geometry_peak"), peak, "geometry peak")
    close(data.get("geometry_range"), spread, "geometry range")
    if (data.get("geometry_signal_active") is not signal
            or data.get("geometry_discriminative") is not discriminative
            or data.get("geometry_active") is not active):
        raise ValueError("V2 signal activation record disagrees with actual reachable candidates")
    reason = data.get("geometry_fallback_reason")
    expected_reason = ("weight_zero" if beta == 0 else "base_all_zero" if not sum(base) > 0
                       else "single_reachable" if len(reachable) == 1 else "weak_geometry" if not signal
                       else "constant_geometry" if not discriminative else "none")
    if reason != expected_reason:
        raise ValueError("V2 fallback reason disagrees with signal application")
    expected_base = normalize(base)
    expected_path = normalize(lengths)
    for offset, index in enumerate(reachable):
        baseline = finite(vectors["baseline_scores"][index], "baseline score")
        if sum(base) > 0:
            close(baseline, expected_base[offset] - .5 * expected_path[offset], "baseline formula")
        elif not 0 <= baseline < 1:
            raise ValueError("V2 zero-base fallback is outside the existing uniform random range")
        q = defect[offset] / peak if active else 0.0
        close(vectors["defect_peak_normalized"][index], q, "reachable geometry normalization")
        close(vectors["geometry_bonus"][index], cap * q, "bounded geometry bonus")
        close(vectors["final_scores"][index], baseline + cap * q, "final score")
    for index in set(range(count)) - set(reachable):
        if vectors["baseline_scores"][index] is not None or vectors["final_scores"][index] is not None:
            raise ValueError("Unreachable candidates must have excluded scores")
        if vectors["geometry_bonus"][index] != 0 or vectors["defect_peak_normalized"][index] != 0:
            raise ValueError("Unreachable candidates received a geometry reward")
    baseline_index = max(reachable, key=lambda index: vectors["baseline_scores"][index])
    selected = max(reachable, key=lambda index: vectors["final_scores"][index])
    if data.get("baseline_selected_index") != baseline_index or data.get("selected_index") != selected:
        raise ValueError("V2 candidate selection is not the recorded first argmax")
    regret = vectors["baseline_scores"][baseline_index] - vectors["baseline_scores"][selected]
    close(data.get("baseline_regret"), regret, "actual baseline regret")
    close(data.get("selected_baseline_score"), vectors["baseline_scores"][selected], "selected baseline score")
    recorded_tolerance = finite(data.get("regret_tolerance"), "regret tolerance")
    if (regret > cap + recorded_tolerance or data.get("regret_bound_satisfied") is not True
            or not 0 <= recorded_tolerance <= 3e-6):
        raise ValueError("V2 selected candidate exceeds the actual bounded baseline regret")
    if data.get("selection_changed") is not (selected != baseline_index):
        raise ValueError("V2 selection-change flag disagrees")
    utility_seconds = finite(data.get("utility_seconds"), "utility duration")
    if utility_seconds < 0:
        raise ValueError("V2 utility duration must be nonnegative")
    return {"event": event, "geometry_signal_active": signal, "geometry_discriminative": discriminative,
            "geometry_active": active, "fallback_reason": reason, "selection_changed": selected != baseline_index,
            "baseline_regret": regret, "bonus_cap": cap, "geometry_peak": peak,
            "utility_seconds": utility_seconds}


def audit_guarded_run(experiment, protocol):
    method = protocol["method"]
    if method not in GUARDED_METHODS:
        return None
    events = protocol["cost"]["events"]
    prefix = protocol["protocol"]["prefix"]
    folder = Path(experiment) / "diagnostics"
    expected = {f"step_{event:04d}.json" for event in range(prefix + 1, events + 1)}
    if {path.name for path in folder.glob("step_*.json")} != expected:
        raise ValueError("The complete, exact v2 post-prefix diagnostic chain is required")
    rows = [audit_guarded_event(json.loads((folder / f"step_{event:04d}.json").read_text(encoding="utf-8")),
                                method, event) for event in range(prefix + 1, events + 1)]
    return {"validated_events": len(rows), "signal_active_events": sum(row["geometry_signal_active"] for row in rows),
            "discriminative_events": sum(row["geometry_discriminative"] for row in rows),
            "reward_applied_events": sum(row["geometry_active"] for row in rows),
            "selection_changed_events": sum(row["selection_changed"] for row in rows),
            "fallback_counts": {reason: sum(row["fallback_reason"] == reason for row in rows)
                                for reason in optimization_recipe("optimization-v2")["fallback"]},
            "maximum_baseline_regret": max((row["baseline_regret"] for row in rows), default=0),
            "baseline_regret_sum": math.fsum(row["baseline_regret"] for row in rows),
            "utility_seconds": math.fsum(row["utility_seconds"] for row in rows), "events": rows}
