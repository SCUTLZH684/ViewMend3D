"""Plot the audited public v1 snapshot; optional matplotlib, no CUDA imports."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.snapshot.read_text(encoding='utf-8'))
    if (data.get('version') != 'viewmend-v1-summary-v1' or data.get('seeds') != [0, 1, 2]
            or data.get('evidence', {}).get('quality_runs') != 24):
        raise ValueError('An audited three-seed v1 snapshot is required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [('accuracy_cm', 'Accuracy (cm), lower is better'),
               ('completion_cm', 'Completion (cm), lower is better'),
               ('coverage_percent', '2 cm coverage (%), higher is better'),
               ('chamfer_mm', 'Chamfer (mm), lower is better')]
    labels = {'confidence_nooracle': 'C', 'random_matched': 'R', 'defect': 'D',
              'defect_no_gate': 'Dng', 'refine_only': 'Ref'}
    colors = {'confidence_nooracle': '#3565bb', 'random_matched': '#78859b', 'defect': '#7062d6',
              'defect_no_gate': '#d98733', 'refine_only': '#2e9581'}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.3), layout='constrained')
    for row, stage in enumerate(('observations', 'time')):
        result = data['stages'][stage]
        methods = list(result['method_statistics'])
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            for x, method in enumerate(methods):
                stat = result['method_statistics'][method]['metrics'][metric]
                if stat['n'] != 3:
                    raise ValueError('Three paired seeds required')
                ax.errorbar(x, stat['mean'], yerr=stat['sample_std'], marker='s',
                            color=colors[method], capsize=4, markersize=6, linewidth=1.7)
                runs = sorted([r for r in result['per_run'] if r['method'] == method], key=lambda r: r['seed'])
                if [r['seed'] for r in runs] != [0, 1, 2]:
                    raise ValueError('Incomplete per-seed records')
                for seed, run in enumerate(runs):
                    ax.scatter(x + (seed - 1) * .085, run['metrics'][metric],
                               s=20, marker=('o', '^', 'v')[seed], color=colors[method],
                               edgecolors='white', linewidths=.5, zorder=3)
            ax.set_xticks(range(len(methods)), [labels[m] for m in methods])
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('60 update events' if stage == 'observations' else '180 s mission budget', fontsize=10)
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=9)
    fig.suptitle('Replica office0 / three paired seeds\nPoints: individual seeds; squares and bars: mean +/- sample SD (not confidence intervals)', fontsize=12)
    fig.supxlabel('C: Confidence without future mask    R: Random matched    D: Defect    Dng: Defect without depth gate    Ref: refine only (20 observations)', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    svg = args.output.with_suffix('.svg')
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)
    print(json.dumps({'output': str(args.output), 'quality_runs': 24}))


if __name__ == '__main__':
    main()
