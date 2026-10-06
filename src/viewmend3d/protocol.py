"""Small, dependency-free definitions for the controlled benchmark protocol."""
from dataclasses import dataclass
import copy
import hashlib
import json


V1_METHODS = ("confidence_nooracle", "random_matched", "defect", "defect_no_gate", "refine_only")
GUARDED_METHODS = ("defect_guarded", "defect_guarded_no_gate")
METHODS = V1_METHODS + GUARDED_METHODS
PROTOCOL_VERSION = "viewmend-observed-only-v1"
V2_PROTOCOL_VERSION = "viewmend-observed-only-v2"
RECIPES = ("optimization-v1", "optimization-v2")


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def optimization_recipe(name):
    """Dependency-free, complete fixed recipe for v2; v1 records stay unchanged."""
    if name != "optimization-v2":
        raise ValueError("A complete recipe is defined only for optimization-v2")
    thresholds = {"opacity": 0.7, "normal_norm": 0.5, "residual_deadzone": 0.05,
                  "depth_jump_absolute": 0.05, "depth_jump_relative": 0.05,
                  "min_valid_fraction": 0.01, "minimum_peak": 1e-4}
    return copy.deepcopy({
        "version": "viewmend-scoring-recipe-v2", "path_length_factor": 0.5,
        "candidate_count": 100, "roi_count": 30, "mapping_optimizer_steps_per_event": 10,
        "render_resolution": [128, 128], "render_ratio": 0.25, "explore_weight": 1000.0,
        "geometry_thresholds": thresholds,
        "methods": {
            "confidence_nooracle": {"scoring_version": "sum_normalized_baseline_v1", "geometry_backend": "none"},
            "defect": {"scoring_version": "sum_normalized_geometry_v1", "geometry_weight": 0.5,
                       "geometry_backend": "per_view", "use_depth_gate": True},
            "defect_guarded": {"scoring_version": "bounded_geometry_v2", "geometry_beta": 0.1,
                               "geometry_backend": "batched", "use_depth_gate": True},
            "defect_guarded_no_gate": {"scoring_version": "bounded_geometry_v2", "geometry_beta": 0.1,
                                       "geometry_backend": "batched", "use_depth_gate": False}},
        "geometry_reward": "beta/reachable_count * clean(defect)/max_reachable_defect; no second normalization",
        "fallback": ["weight_zero", "base_all_zero", "single_reachable", "weak_geometry", "constant_geometry"],
        "future_candidate_observations": False, "random_domain_version": PROTOCOL_VERSION})


def validate_recipe_record(record):
    """Reject altered/missing v2 recipe fields before any output is trusted."""
    version = record.get("version")
    if version == PROTOCOL_VERSION:
        if any(key in record for key in ("recipe", "recipe_spec", "recipe_sha256", "campaign_spec_sha256")):
            raise ValueError("v1 protocol must not contain unversioned v2 recipe fields")
        return
    if version != V2_PROTOCOL_VERSION or record.get("recipe") != "optimization-v2":
        raise ValueError("Unsupported controlled protocol or recipe")
    expected = optimization_recipe("optimization-v2")
    if record.get("recipe_spec") != expected or record.get("recipe_sha256") != canonical_sha256(expected):
        raise ValueError("The complete fixed v2 scoring recipe is missing or changed")
    from .campaign_profiles import campaign_spec_sha256
    if record.get("campaign_spec_sha256") != campaign_spec_sha256("optimization-v2"):
        raise ValueError("The fixed v2 campaign specification is missing or changed")


def domain_seed(seed, domain, event):
    """Stable uint32 seeds; independent of Python's randomized hash()."""
    value = f"{PROTOCOL_VERSION}:{seed}:{domain}:{event}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


@dataclass(frozen=True)
class Protocol:
    observations: int = 60
    prefix: int = 20
    checkpoint_every: int = 20
    mode: str = "observations"
    seconds: float = 180.0
    candidate_count: int = 100
    roi_count: int = 30
    sample_points: int = 500000
    safety_event_cap: int = 10000
    recipe: str = "optimization-v1"
    campaign_spec_sha256: str = None

    def validate(self):
        if self.recipe not in RECIPES:
            raise ValueError("Unsupported optimization recipe")
        if self.recipe == "optimization-v1" and self.campaign_spec_sha256 is not None:
            raise ValueError("A v2 campaign specification cannot be attached to v1")
        if self.recipe == "optimization-v2":
            validate_recipe_record(self.as_dict())
            if self.candidate_count != 100 or self.roi_count != 30 or self.sample_points != 500000:
                raise ValueError("v2 fixes candidate/ROI counts and evaluation sample points")
        for name in ("observations", "prefix", "checkpoint_every", "candidate_count", "sample_points", "safety_event_cap"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.prefix >= self.observations:
            raise ValueError("prefix must be smaller than observations")
        if self.safety_event_cap <= self.prefix:
            raise ValueError("safety_event_cap must exceed prefix")
        if self.mode not in ("observations", "time"):
            raise ValueError("mode must be observations or time")
        if self.seconds <= 0 or self.seconds == float("inf") or self.seconds != self.seconds:
            raise ValueError("seconds must be finite and positive")
        if type(self.roi_count) is not int or not 0 <= self.roi_count <= self.candidate_count:
            raise ValueError("roi_count must be between 0 and candidate_count")
        return self

    @property
    def checkpoints(self):
        if self.mode == "time":
            return (self.prefix,)
        middle = (self.prefix + self.observations) // 2
        scheduled = range(self.checkpoint_every, self.observations + 1, self.checkpoint_every)
        return tuple(sorted({self.prefix, middle, self.observations,
                             *(event for event in scheduled if event >= self.prefix)}))

    def as_dict(self):
        values = {key: value for key, value in self.__dict__.items()
                  if key not in ("recipe", "campaign_spec_sha256")}
        record = {**values, "version": PROTOCOL_VERSION,
                "checkpoints": list(self.checkpoints),
                "time_checkpoints_seconds": [self.seconds / 3, self.seconds * 2 / 3, self.seconds]
                                            if self.mode == "time" else [],
                "fixed_observation_budget": self.observations if self.mode == "observations" else None,
                "future_candidate_depth_mask": False,
                "shared_scene_bounds": True,
                "mapping_optimizer_steps_per_event": 10,
                "sampling_domains": ["planning", "sensor", "mapping", "evaluation"],
                "time_definition": "synchronized planning + mapping + simulated flight; sensor and wall time separate"}
        if self.recipe == "optimization-v2":
            spec = optimization_recipe(self.recipe)
            record.update(version=V2_PROTOCOL_VERSION, recipe=self.recipe,
                          recipe_spec=spec, recipe_sha256=canonical_sha256(spec),
                          campaign_spec_sha256=self.campaign_spec_sha256)
        return record

    @property
    def event_cap(self):
        return self.safety_event_cap if self.mode == "time" else self.observations
