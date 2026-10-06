"""Trusted, dependency-free campaign plans; no GPU or file-system inspection.

Profiles are selected by an explicit registered name. The v1 default remains
unchanged; v2 includes the complete recipe in its frozen campaign fingerprint.
"""
import hashlib
import json

from .protocol import optimization_recipe

DEFAULT_CAMPAIGN = "optimization-v1"
CAMPAIGN_NAMES = (DEFAULT_CAMPAIGN, "optimization-v2")


def get_profile(name=DEFAULT_CAMPAIGN):
    if name not in CAMPAIGN_NAMES:
        raise ValueError(f"Unregistered campaign: {name}")
    if name == DEFAULT_CAMPAIGN:
        main = ["confidence_nooracle", "random_matched", "defect"]
        stages = [
            {"name": "smoke", "methods": ["confidence_nooracle", "defect", "refine_only"],
             "seeds": [0], "frames": 6, "prefix": 2, "mode": "observations", "seconds": 180},
            {"name": "observations", "methods": main + ["defect_no_gate", "refine_only"],
             "seeds": [0, 1, 2], "frames": 60, "prefix": 20, "mode": "observations", "seconds": 180},
            {"name": "time", "methods": main, "seeds": [0, 1, 2],
             "frames": 60, "prefix": 20, "mode": "time", "seconds": 180},
        ]
        version = "viewmend-campaign-v1"
    else:
        main = ["confidence_nooracle", "defect_guarded"]
        stages = [
            {"name": "smoke", "methods": main + ["defect_guarded_no_gate"],
             "seeds": [0], "frames": 6, "prefix": 2, "mode": "observations", "seconds": 180},
            {"name": "development_observations", "methods": ["confidence_nooracle", "defect", "defect_guarded", "defect_guarded_no_gate"],
             "seeds": [0, 1], "frames": 60, "prefix": 20, "mode": "observations", "seconds": 180},
            {"name": "development_time", "methods": main,
             "seeds": [0, 1], "frames": 60, "prefix": 20, "mode": "time", "seconds": 180},
            {"name": "heldout_observations", "methods": main,
             "seeds": [3, 4, 5], "frames": 60, "prefix": 20, "mode": "observations", "seconds": 180},
            {"name": "heldout_time", "methods": main,
             "seeds": [3, 4, 5], "frames": 60, "prefix": 20, "mode": "time", "seconds": 180},
        ]
        version = "viewmend-campaign-v2"
    profile = {"name": name, "version": version, "scene": "replica/office0", "stages": stages}
    if name != DEFAULT_CAMPAIGN:
        profile.update(recipe=name, recipe_spec=optimization_recipe(name),
                       candidate_count=100, roi_count=30, sample_points=500000,
                       checkpoint_every=20, safety_event_cap=10000,
                       idle_limits={"memory_mb_below": 1024, "utilization_max": 5},
                       failure_policy="stop; preserve logs and outputs; no automatic retries",
                       export_smoke=True)
    # Never expose shared mutable list/dict objects to callers.
    return json.loads(json.dumps(profile, allow_nan=False))


def campaign_spec_sha256(name=DEFAULT_CAMPAIGN):
    encoded = json.dumps(get_profile(name), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_profile_state(state, name=DEFAULT_CAMPAIGN):
    profile = get_profile(name)
    if (not isinstance(state, dict) or state.get("version") != profile["version"]
            or type(state.get("stage_index")) is not int
            or not 0 <= state["stage_index"] <= len(profile["stages"])):
        raise ValueError("Unsupported or malformed campaign state; inspect it manually")
    if name != DEFAULT_CAMPAIGN:
        if (state.get("campaign") != name
                or state.get("campaign_spec_sha256") != campaign_spec_sha256(name)
                or state.get("campaign_spec") != profile):
            raise ValueError("Frozen campaign specification changed or is missing; inspect it manually")
        status, index = state.get("status"), state["stage_index"]
        if (status not in ("not_started", "waiting", "running", "failed", "completed")
                or (status == "completed") != (index == len(profile["stages"]))
                or (status == "not_started" and index != 0)):
            raise ValueError("Campaign status and registered stage disagree")
        attempts, active = state.get("attempts"), state.get("active")
        if not isinstance(attempts, list) or (active is not None and not isinstance(active, dict)):
            raise ValueError("Invalid campaign attempt records")
        completed, tokens = {}, set()
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise ValueError("Invalid campaign attempt record")
            step, token = attempt.get("stage_index"), attempt.get("token")
            if (type(step) is not int or not 0 <= step < len(profile["stages"])
                    or attempt.get("stage") != profile["stages"][step]["name"]
                    or attempt.get("status") not in ("starting", "running", "deferred", "failed", "completed")
                    or not isinstance(token, str) or not token or token in tokens):
                raise ValueError("Invalid or duplicate campaign attempt identity")
            tokens.add(token)
            if attempt["status"] == "completed":
                expected = len(profile["stages"][step]["methods"]) * len(profile["stages"][step]["seeds"])
                if (step in completed or type(attempt.get("validated_experiments")) is not int
                        or attempt["validated_experiments"] != expected):
                    raise ValueError("Invalid validated campaign stage count")
                completed[step] = expected
        if set(completed) != set(range(index)):
            raise ValueError("Campaign stage lacks its complete validation history")
        if active is not None:
            if active.get("token") not in tokens or not any(attempt == active for attempt in attempts):
                raise ValueError("Active campaign attempt disagrees with its durable history")
        if status == "running" and (not active or active.get("stage_index") != index
                or active.get("status") not in ("starting", "running")):
            raise ValueError("Running campaign lacks a registered active attempt")
    return profile
