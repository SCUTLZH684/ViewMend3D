"""CPU figures from a trusted complete v2 analysis; no Torch/CUDA imports."""
from pathlib import Path

COLORS = {"confidence_nooracle": "#3565bb", "defect": "#92929d",
          "defect_guarded": "#7062d6", "defect_guarded_no_gate": "#d98733"}
SHORT = {"confidence_nooracle": "C", "defect": "D1", "defect_guarded": "G2", "defect_guarded_no_gate": "G2ng"}
STAGES = ("development_observations", "development_time", "heldout_observations", "heldout_time")


def draw(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, family, metrics in (
        ("quality", "metrics", (("accuracy_cm", "Accuracy (cm), lower"), ("completion_cm", "Completion (cm), lower"),
                                ("coverage_percent", "2 cm coverage (%), higher"), ("chamfer_mm", "Chamfer (mm), lower"))),
        ("cost", "cost", (("planning_seconds", "Planning (s)"), ("wall_seconds", "Reconstruction wall (s)"),
                          ("observations", "Acquired observations"), ("peak_torch_allocated_mb", "Torch peak (MiB)")))):
        fig, axes = plt.subplots(4, 4, figsize=(14, 12), layout="constrained")
        for row, stage_name in enumerate(STAGES):
            stage = data["stages"][stage_name]
            methods = list(stage["method_statistics"])
            seeds = stage["seeds"]
            for col, (metric, title) in enumerate(metrics):
                ax = axes[row, col]
                for x, method in enumerate(methods):
                    stat = stage["method_statistics"][method][family][metric]
                    ax.errorbar(x, stat["mean"], yerr=stat["sample_std"], marker="s", capsize=4,
                                color=COLORS[method], linewidth=1.4, markersize=5)
                    runs = {r["seed"]: r for r in stage["per_run"] if r["method"] == method}
                    for i, seed in enumerate(seeds):
                        ax.scatter(x + (i - (len(seeds)-1)/2) * .1, runs[seed][family][metric],
                                   marker=("o", "^", "v")[i], color=COLORS[method], edgecolors="white", linewidths=.4, s=26, zorder=3)
                ax.set_xticks(range(len(methods)), [SHORT[m] for m in methods])
                ax.set_title(title, fontsize=10)
                partition = "Development n=2" if row < 2 else "Held-out seeds n=3"
                budget = "60 observations/updates" if row % 2 == 0 else "180 s mission"
                ax.set_xlabel(f"{partition}\n{budget}", fontsize=9)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", alpha=.2)
                ax.tick_params(labelsize=8)
        fig.suptitle("Replica office0 / four separate paired groups\nPoints: seeds; squares/bars: mean +/- sample SD (not CI)", fontsize=12)
        fig.supxlabel("C: Confidence; D1: legacy Defect rerun; G2: preregistered primary; G2ng: no-gate ablation. No pooled result.", fontsize=9)
        for extension in ("png", "svg"):
            with (Path(output) / f"{name}.{extension}").open("xb") as stream:
                fig.savefig(stream, format=extension, dpi=160)
        plt.close(fig)


def main(argv=None):
    import argparse
    import json
    from publish_v2_results import DEFAULT_SCHEMA, load_verified_analysis, safe_output_directory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path, help="与发布工具相同的可信全27分支最终分析 JSON")
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, required=True, help="新的 setup/runs 子目录或仓库外私人目录")
    args = parser.parse_args(argv)
    try:
        _, result = load_verified_analysis(args.analysis, args.expected_sha256, args.schema)
        target = safe_output_directory(args.output_dir)
        import matplotlib
        matplotlib.use("Agg")
        target.mkdir(parents=True)
        draw(result, target)
        print(json.dumps({"status": "complete", "plots_generated": True, "output_dir": str(target),
                          "analysis_sha256": result["evidence"]["analysis_sha256"], "validated_branches": 27}))
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        print(json.dumps({"status": "rejected", "plots_generated": False, "reason": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
