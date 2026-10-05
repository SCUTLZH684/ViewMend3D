"""Small, dependency-free definitions for the controlled benchmark protocol."""
from dataclasses import dataclass
import hashlib


METHODS = ("confidence_nooracle", "random_matched", "defect", "defect_no_gate", "refine_only")
PROTOCOL_VERSION = "viewmend-observed-only-v1"


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

    def validate(self):
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
        return {**self.__dict__, "version": PROTOCOL_VERSION,
                "checkpoints": list(self.checkpoints),
                "time_checkpoints_seconds": [self.seconds / 3, self.seconds * 2 / 3, self.seconds]
                                            if self.mode == "time" else [],
                "fixed_observation_budget": self.observations if self.mode == "observations" else None,
                "future_candidate_depth_mask": False,
                "shared_scene_bounds": True,
                "mapping_optimizer_steps_per_event": 10,
                "sampling_domains": ["planning", "sensor", "mapping", "evaluation"],
                "time_definition": "synchronized planning + mapping + simulated flight; sensor and wall time separate"}

    @property
    def event_cap(self):
        return self.safety_event_cap if self.mode == "time" else self.observations
