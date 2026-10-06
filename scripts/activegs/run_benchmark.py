"""CLI for observed-only, paired ViewMend3D experiments; --dry-run needs no GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from viewmend3d.protocol import METHODS, V1_METHODS, Protocol, optimization_recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scene", default="replica/office0")
    parser.add_argument("--methods", nargs="+", default=["confidence_nooracle", "random_matched", "defect"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--frames", type=int, default=60, help="Observation/update-event cap; use a large cap for time mode")
    parser.add_argument("--prefix-frames", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--protocol", choices=("observations", "time"), default="observations")
    parser.add_argument("--budget", type=float, default=180.0)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--roi-count", type=int, default=30)
    parser.add_argument("--sample-points", type=int, default=500000)
    parser.add_argument("--prefix-cache-dir", type=Path)
    parser.add_argument("--recipe", choices=("optimization-v1", "optimization-v2"), default="optimization-v1")
    parser.add_argument("--campaign-spec-sha256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        methods = tuple(args.methods)
        seeds = tuple(args.seeds)
        if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
            raise ValueError(f"methods must be unique values from {METHODS}")
        allowed = V1_METHODS if args.recipe == "optimization-v1" else optimization_recipe(args.recipe)["methods"]
        if not set(methods) <= set(allowed):
            raise ValueError("Methods do not belong to the selected versioned recipe")
        if args.recipe == "optimization-v2" and args.scene != "replica/office0":
            raise ValueError("The preregistered v2 campaign fixes scene replica/office0")
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be unique nonnegative integers")
        protocol = Protocol(observations=args.frames, prefix=args.prefix_frames,
                            checkpoint_every=args.checkpoint_every, mode=args.protocol, seconds=args.budget,
                            candidate_count=args.candidate_count, roi_count=args.roi_count,
                            sample_points=args.sample_points, recipe=args.recipe,
                            campaign_spec_sha256=args.campaign_spec_sha256).validate()
        if args.gpu < 0:
            raise ValueError("GPU must be nonnegative")
        if not args.upstream.is_dir() or not (args.upstream / "config/main.yaml").is_file():
            raise ValueError("upstream must be the pinned ActiveGS source directory")
        if not args.run_dir.is_absolute():
            raise ValueError("run-dir must be absolute")
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"protocol": protocol.as_dict(), "methods": methods, "seeds": seeds,
                          "scene": args.scene, "run_dir": str(args.run_dir),
                          "status": "plan_only_no_gpu_allocation"}, ensure_ascii=False, indent=2))
        return
    from viewmend3d.benchmark import run_benchmark
    run_benchmark(args.upstream, args.run_dir, args.gpu, protocol, methods, seeds,
                  scene=args.scene, prefix_cache_dir=args.prefix_cache_dir)


if __name__ == "__main__":
    main()
