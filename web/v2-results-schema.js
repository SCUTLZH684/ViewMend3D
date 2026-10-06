// Frozen preregistration metadata only. No measured or synthetic results.
const frozen = {
  "source_commit": "fafb552400903cd083a34f3966bf5068a95344d2",
  "source_identity": {
    "project_commit": "fafb552400903cd083a34f3966bf5068a95344d2",
    "project_source_sha256": "566fbaba1093a878d63005c907eadba869b7d43770261ea439cc15964eec2388",
    "upstream_commit": "558121a00b2eca84d9851a609e99ebb8e26ea2d8",
    "upstream_source_sha256": "2e2c5cfd6c8cab123da58ddc420fd588d3f08c6b2f6ed07cfce2363079a5fc0f",
    "upstream_diff_sha256": "dae6df308d4b27016270ae5ed40019e0b61d75b930223925cf262361b75ffeba"
  },
  "campaign_spec_sha256": "d3802aa30065a1a38c25dcd0a742440cc0440b8791519c0cf2a489bbfd53f541",
  "profile": {
    "name": "optimization-v2",
    "version": "viewmend-campaign-v2",
    "scene": "replica/office0",
    "stages": [
      {
        "name": "smoke",
        "methods": [
          "confidence_nooracle",
          "defect_guarded",
          "defect_guarded_no_gate"
        ],
        "seeds": [
          0
        ],
        "frames": 6,
        "prefix": 2,
        "mode": "observations",
        "seconds": 180
      },
      {
        "name": "development_observations",
        "methods": [
          "confidence_nooracle",
          "defect",
          "defect_guarded",
          "defect_guarded_no_gate"
        ],
        "seeds": [
          0,
          1
        ],
        "frames": 60,
        "prefix": 20,
        "mode": "observations",
        "seconds": 180
      },
      {
        "name": "development_time",
        "methods": [
          "confidence_nooracle",
          "defect_guarded"
        ],
        "seeds": [
          0,
          1
        ],
        "frames": 60,
        "prefix": 20,
        "mode": "time",
        "seconds": 180
      },
      {
        "name": "heldout_observations",
        "methods": [
          "confidence_nooracle",
          "defect_guarded"
        ],
        "seeds": [
          3,
          4,
          5
        ],
        "frames": 60,
        "prefix": 20,
        "mode": "observations",
        "seconds": 180
      },
      {
        "name": "heldout_time",
        "methods": [
          "confidence_nooracle",
          "defect_guarded"
        ],
        "seeds": [
          3,
          4,
          5
        ],
        "frames": 60,
        "prefix": 20,
        "mode": "time",
        "seconds": 180
      }
    ],
    "recipe": "optimization-v2",
    "recipe_spec": {
      "version": "viewmend-scoring-recipe-v2",
      "path_length_factor": 0.5,
      "candidate_count": 100,
      "roi_count": 30,
      "mapping_optimizer_steps_per_event": 10,
      "render_resolution": [
        128,
        128
      ],
      "render_ratio": 0.25,
      "explore_weight": 1000.0,
      "geometry_thresholds": {
        "opacity": 0.7,
        "normal_norm": 0.5,
        "residual_deadzone": 0.05,
        "depth_jump_absolute": 0.05,
        "depth_jump_relative": 0.05,
        "min_valid_fraction": 0.01,
        "minimum_peak": 0.0001
      },
      "methods": {
        "confidence_nooracle": {
          "scoring_version": "sum_normalized_baseline_v1",
          "geometry_backend": "none"
        },
        "defect": {
          "scoring_version": "sum_normalized_geometry_v1",
          "geometry_weight": 0.5,
          "geometry_backend": "per_view",
          "use_depth_gate": true
        },
        "defect_guarded": {
          "scoring_version": "bounded_geometry_v2",
          "geometry_beta": 0.1,
          "geometry_backend": "batched",
          "use_depth_gate": true
        },
        "defect_guarded_no_gate": {
          "scoring_version": "bounded_geometry_v2",
          "geometry_beta": 0.1,
          "geometry_backend": "batched",
          "use_depth_gate": false
        }
      },
      "geometry_reward": "beta/reachable_count * clean(defect)/max_reachable_defect; no second normalization",
      "fallback": [
        "weight_zero",
        "base_all_zero",
        "single_reachable",
        "weak_geometry",
        "constant_geometry"
      ],
      "future_candidate_observations": false,
      "random_domain_version": "viewmend-observed-only-v1"
    },
    "candidate_count": 100,
    "roi_count": 30,
    "sample_points": 500000,
    "checkpoint_every": 20,
    "safety_event_cap": 10000,
    "idle_limits": {
      "memory_mb_below": 1024,
      "utilization_max": 5
    },
    "failure_policy": "stop; preserve logs and outputs; no automatic retries",
    "export_smoke": true
  },
  "protocols": {
    "smoke": {
      "observations": 6,
      "prefix": 2,
      "checkpoint_every": 20,
      "mode": "observations",
      "seconds": 180,
      "candidate_count": 100,
      "roi_count": 30,
      "sample_points": 500000,
      "safety_event_cap": 10000,
      "version": "viewmend-observed-only-v2",
      "checkpoints": [
        2,
        4,
        6
      ],
      "time_checkpoints_seconds": [],
      "fixed_observation_budget": 6,
      "future_candidate_depth_mask": false,
      "shared_scene_bounds": true,
      "mapping_optimizer_steps_per_event": 10,
      "sampling_domains": [
        "planning",
        "sensor",
        "mapping",
        "evaluation"
      ],
      "time_definition": "synchronized planning + mapping + simulated flight; sensor and wall time separate",
      "recipe": "optimization-v2",
      "recipe_spec": {
        "version": "viewmend-scoring-recipe-v2",
        "path_length_factor": 0.5,
        "candidate_count": 100,
        "roi_count": 30,
        "mapping_optimizer_steps_per_event": 10,
        "render_resolution": [
          128,
          128
        ],
        "render_ratio": 0.25,
        "explore_weight": 1000.0,
        "geometry_thresholds": {
          "opacity": 0.7,
          "normal_norm": 0.5,
          "residual_deadzone": 0.05,
          "depth_jump_absolute": 0.05,
          "depth_jump_relative": 0.05,
          "min_valid_fraction": 0.01,
          "minimum_peak": 0.0001
        },
        "methods": {
          "confidence_nooracle": {
            "scoring_version": "sum_normalized_baseline_v1",
            "geometry_backend": "none"
          },
          "defect": {
            "scoring_version": "sum_normalized_geometry_v1",
            "geometry_weight": 0.5,
            "geometry_backend": "per_view",
            "use_depth_gate": true
          },
          "defect_guarded": {
            "scoring_version": "bounded_geometry_v2",
            "geometry_beta": 0.1,
            "geometry_backend": "batched",
            "use_depth_gate": true
          },
          "defect_guarded_no_gate": {
            "scoring_version": "bounded_geometry_v2",
            "geometry_beta": 0.1,
            "geometry_backend": "batched",
            "use_depth_gate": false
          }
        },
        "geometry_reward": "beta/reachable_count * clean(defect)/max_reachable_defect; no second normalization",
        "fallback": [
          "weight_zero",
          "base_all_zero",
          "single_reachable",
          "weak_geometry",
          "constant_geometry"
        ],
        "future_candidate_observations": false,
        "random_domain_version": "viewmend-observed-only-v1"
      },
      "recipe_sha256": "13dc1b2b721d3129b74e3046bd02731cd8c60950c2ee45f8d6842b0e692cd745",
      "campaign_spec_sha256": "d3802aa30065a1a38c25dcd0a742440cc0440b8791519c0cf2a489bbfd53f541"
    },
    "development_observations": {
      "observations": 60,
      "prefix": 20,
      "checkpoint_every": 20,
      "mode": "observations",
      "seconds": 180,
      "candidate_count": 100,
      "roi_count": 30,
      "sample_points": 500000,
      "safety_event_cap": 10000,
      "version": "viewmend-observed-only-v2",
      "checkpoints": [
        20,
        40,
        60
      ],
      "time_checkpoints_seconds": [],
      "fixed_observation_budget": 60,
      "future_candidate_depth_mask": false,
      "shared_scene_bounds": true,
      "mapping_optimizer_steps_per_event": 10,
      "sampling_domains": [
        "planning",
        "sensor",
        "mapping",
        "evaluation"
      ],
      "time_definition": "synchronized planning + mapping + simulated flight; sensor and wall time separate",
      "recipe": "optimization-v2",
      "recipe_spec": {
        "version": "viewmend-scoring-recipe-v2",
        "path_length_factor": 0.5,
        "candidate_count": 100,
        "roi_count": 30,
        "mapping_optimizer_steps_per_event": 10,
        "render_resolution": [
          128,
          128
        ],
        "render_ratio": 0.25,
        "explore_weight": 1000.0,
        "geometry_thresholds": {
          "opacity": 0.7,
          "normal_norm": 0.5,
          "residual_deadzone": 0.05,
          "depth_jump_absolute": 0.05,
          "depth_jump_relative": 0.05,
          "min_valid_fraction": 0.01,
          "minimum_peak": 0.0001
        },
        "methods": {
          "confidence_nooracle": {
            "scoring_version": "sum_normalized_baseline_v1",
            "geometry_backend": "none"
          },
          "defect": {
            "scoring_version": "sum_normalized_geometry_v1",
            "geometry_weight": 0.5,
            "geometry_backend": "per_view",
            "use_depth_gate": true
          },
          "defect_guarded": {
            "scoring_version": "bounded_geometry_v2",
            "geometry_beta": 0.1,
            "geometry_backend": "batched",
            "use_depth_gate": true
          },
          "defect_guarded_no_gate": {
            "scoring_version": "bounded_geometry_v2",
            "geometry_beta": 0.1,
            "geometry_backend": "batched",
            "use_depth_gate": false
          }
        },
        "geometry_reward": "beta/reachable_count * clean(defect)/max_reachable_defect; no second normalization",
        "fallback": [
          "weight_zero",
          "base_all_zero",
          "single_reachable",
          "weak_geometry",
          "constant_geometry"
        ],
        "future_candidate_observations": false,
        "random_domain_version": "viewmend-observed-only-v1"
      },
      "recipe_sha256": "13dc1b2b721d3129b74e3046bd02731cd8c60950c2ee45f8d6842b0e692cd745",
      "campaign_spec_sha256": "d3802aa30065a1a38c25dcd0a742440cc0440b8791519c0cf2a489bbfd53f541"
    },
    "development_time": {
      "observations": 60,
      "prefix": 20,
      "checkpoint_every": 20,
      "mode": "time",
      "seconds": 180,
      "candidate_count": 100,
      "roi_count": 30,
      "sample_points": 500000,
      "safety_event_cap": 10000,
      "version": "viewmend-observed-only-v2",
      "checkpoints": [
        20
      ],
      "time_checkpoints_seconds": [
        60.0,
        120.0,
        180
      ],
      "fixed_observation_budget": null,
      "future_candidate_depth_mask": false,
      "shared_scene_bounds": true,
      "mapping_optimizer_steps_per_event": 10,
      "sampling_domains": [
        "planning",
        "sensor",
        "mapping",
        "evaluation"
      ],
      "time_definition": "synchronized planning + mapping + simulated flight; sensor and wall time separate",
      "recipe": "optimization-v2",
      "recipe_spec": {
        "version": "viewmend-scoring-recipe-v2",
        "path_length_factor": 0.5,
        "candidate_count": 100,
        "roi_count": 30,
        "mapping_optimizer_steps_per_event": 10,
        "render_resolution": [
          128,
          128
        ],
        "render_ratio": 0.25,
        "explore_weight": 1000.0,
        "geometry_thresholds": {
          "opacity": 0.7,
          "normal_norm": 0.5,
          "residual_deadzone": 0.05,
          "depth_jump_absolute": 0.05,
          "depth_jump_relative": 0.05,
          "min_valid_fraction": 0.01,
          "minimum_peak": 0.0001
        },
        "methods": {
          "confidence_nooracle": {
            "scoring_version": "sum_normalized_baseline_v1",
            "geometry_backend": "none"
          },
          "defect": {
            "scoring_version": "sum_normalized_geometry_v1",
            "geometry_weight": 0.5,
            "geometry_backend": "per_view",
            "use_depth_gate": true
          },
          "defect_guarded": {
            "scoring_version": "bounded_geometry_v2",
            "geometry_beta": 0.1,
            "geometry_backend": "batched",
            "use_depth_gate": true
          },
          "defect_guarded_no_gate": {
            "scoring_version": "bounded_geometry_v2",
            "geometry_beta": 0.1,
            "geometry_backend": "batched",
            "use_depth_gate": false
          }
        },
        "geometry_reward": "beta/reachable_count * clean(defect)/max_reachable_defect; no second normalization",
        "fallback": [
          "weight_zero",
          "base_all_zero",
          "single_reachable",
          "weak_geometry",
          "constant_geometry"
        ],
        "future_candidate_observations": false,
        "random_domain_version": "viewmend-observed-only-v1"
      },
      "recipe_sha256": "13dc1b2b721d3129b74e3046bd02731cd8c60950c2ee45f8d6842b0e692cd745",
      "campaign_spec_sha256": "d3802aa30065a1a38c25dcd0a742440cc0440b8791519c0cf2a489bbfd53f541"
    },
    "heldout_observations": {
      "observations": 60,
      "prefix": 20,
      "checkpoint_every": 20,
      "mode": "observations",
      "seconds": 180,
      "candidate_count": 100,
      "roi_count": 30,
      "sample_points": 500000,
      "safety_event_cap": 10000,
      "version": "viewmend-observed-only-v2",
      "checkpoints": [
        20,
        40,
        60
      ],
      "time_checkpoints_seconds": [],
      "fixed_observation_budget": 60,
      "future_candidate_depth_mask": false,
      "shared_scene_bounds": true,
      "mapping_optimizer_steps_per_event": 10,
      "sampling_domains": [
        "planning",
        "sensor",
        "mapping",
        "evaluation"
      ],
      "time_definition": "synchronized planning + mapping + simulated flight; sensor and wall time separate",
      "recipe": "optimization-v2",
      "recipe_spec": {
        "version": "viewmend-scoring-recipe-v2",
        "path_length_factor": 0.5,
        "candidate_count": 100,
        "roi_count": 30,
        "mapping_optimizer_steps_per_event": 10,
        "render_resolution": [
          128,
          128
        ],
        "render_ratio": 0.25,
        "explore_weight": 1000.0,
        "geometry_thresholds": {
          "opacity": 0.7,
          "normal_norm": 0.5,
          "residual_deadzone": 0.05,
          "depth_jump_absolute": 0.05,
          "depth_jump_relative": 0.05,
          "min_valid_fraction": 0.01,
          "minimum_peak": 0.0001
        },
        "methods": {
          "confidence_nooracle": {
            "scoring_version": "sum_normalized_baseline_v1",
            "geometry_backend": "none"
          },
          "defect": {
            "scoring_version": "sum_normalized_geometry_v1",
            "geometry_weight": 0.5,
            "geometry_backend": "per_view",
            "use_depth_gate": true
          },
          "defect_guarded": {
            "scoring_version": "bounded_geometry_v2",
            "geometry_beta": 0.1,
            "geometry_backend": "batched",
            "use_depth_gate": true
          },
          "defect_guarded_no_gate": {
            "scoring_version": "bounded_geometry_v2",
            "geometry_beta": 0.1,
            "geometry_backend": "batched",
            "use_depth_gate": false
          }
        },
        "geometry_reward": "beta/reachable_count * clean(defect)/max_reachable_defect; no second normalization",
        "fallback": [
          "weight_zero",
          "base_all_zero",
          "single_reachable",
          "weak_geometry",
          "constant_geometry"
        ],
        "future_candidate_observations": false,
        "random_domain_version": "viewmend-observed-only-v1"
      },
      "recipe_sha256": "13dc1b2b721d3129b74e3046bd02731cd8c60950c2ee45f8d6842b0e692cd745",
      "campaign_spec_sha256": "d3802aa30065a1a38c25dcd0a742440cc0440b8791519c0cf2a489bbfd53f541"
    },
    "heldout_time": {
      "observations": 60,
      "prefix": 20,
      "checkpoint_every": 20,
      "mode": "time",
      "seconds": 180,
      "candidate_count": 100,
      "roi_count": 30,
      "sample_points": 500000,
      "safety_event_cap": 10000,
      "version": "viewmend-observed-only-v2",
      "checkpoints": [
        20
      ],
      "time_checkpoints_seconds": [
        60.0,
        120.0,
        180
      ],
      "fixed_observation_budget": null,
      "future_candidate_depth_mask": false,
      "shared_scene_bounds": true,
      "mapping_optimizer_steps_per_event": 10,
      "sampling_domains": [
        "planning",
        "sensor",
        "mapping",
        "evaluation"
      ],
      "time_definition": "synchronized planning + mapping + simulated flight; sensor and wall time separate",
      "recipe": "optimization-v2",
      "recipe_spec": {
        "version": "viewmend-scoring-recipe-v2",
        "path_length_factor": 0.5,
        "candidate_count": 100,
        "roi_count": 30,
        "mapping_optimizer_steps_per_event": 10,
        "render_resolution": [
          128,
          128
        ],
        "render_ratio": 0.25,
        "explore_weight": 1000.0,
        "geometry_thresholds": {
          "opacity": 0.7,
          "normal_norm": 0.5,
          "residual_deadzone": 0.05,
          "depth_jump_absolute": 0.05,
          "depth_jump_relative": 0.05,
          "min_valid_fraction": 0.01,
          "minimum_peak": 0.0001
        },
        "methods": {
          "confidence_nooracle": {
            "scoring_version": "sum_normalized_baseline_v1",
            "geometry_backend": "none"
          },
          "defect": {
            "scoring_version": "sum_normalized_geometry_v1",
            "geometry_weight": 0.5,
            "geometry_backend": "per_view",
            "use_depth_gate": true
          },
          "defect_guarded": {
            "scoring_version": "bounded_geometry_v2",
            "geometry_beta": 0.1,
            "geometry_backend": "batched",
            "use_depth_gate": true
          },
          "defect_guarded_no_gate": {
            "scoring_version": "bounded_geometry_v2",
            "geometry_beta": 0.1,
            "geometry_backend": "batched",
            "use_depth_gate": false
          }
        },
        "geometry_reward": "beta/reachable_count * clean(defect)/max_reachable_defect; no second normalization",
        "fallback": [
          "weight_zero",
          "base_all_zero",
          "single_reachable",
          "weak_geometry",
          "constant_geometry"
        ],
        "future_candidate_observations": false,
        "random_domain_version": "viewmend-observed-only-v1"
      },
      "recipe_sha256": "13dc1b2b721d3129b74e3046bd02731cd8c60950c2ee45f8d6842b0e692cd745",
      "campaign_spec_sha256": "d3802aa30065a1a38c25dcd0a742440cc0440b8791519c0cf2a489bbfd53f541"
    }
  },
  "schema_sha256": "fd10438775a477a700c15a5d7d8c359700f240b9392ae2673106b17557add34b"
};
function freeze(value) { if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); } return value; }
export const FROZEN = freeze(frozen);
