"""Analysis guard/arithmetic checks; synthetic fixtures are not results."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

source = Path(__file__).resolve().parents[1] / 'scripts/activegs/analyze_campaign.py'
spec = importlib.util.spec_from_file_location('campaign_analysis', source)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
assert analysis.stats([1, 2, 3]) == {'n': 3, 'mean': 2, 'sample_std': 1.0}
assert analysis.stats([]) == {'n': 0, 'mean': None, 'sample_std': None}
assert analysis.normalize([0, 0]) == [0, 0]
assert analysis.ranking([1, 1, 0]) == [0, 1, 2]
runs = [{'method': method, 'seed': seed, 'metrics': {'test': seed + adjustment}}
        for method, adjustment in [('confidence_nooracle', 0), ('defect', 1)] for seed in (0, 1, 2)]
paired = analysis.paired_statistics(runs, 'defect', 'confidence_nooracle', {'metrics': ['test']})
assert paired['metrics']['test']['mean'] == 1
assert paired['metrics']['test']['sample_std'] == 0
assert paired['metrics']['test']['relative_percent_by_seed']['0'] is None
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    state = root / 'runs/campaigns/optimization-v1/state.json'
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'version': 'viewmend-campaign-v1', 'status': 'running', 'stage_index': 1}))
    target = root / 'result.json'
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = analysis.main(['--root', str(root), '--output', str(target)])
    report = json.loads(output.getvalue())
    assert code == 2 and report['status'] == 'pending'
    assert report['quality_summary_generated'] is False and not target.exists()
    stage = root / 'stage'
    stage.mkdir()
    (stage / 'benchmark-summary.json').write_text(json.dumps({'status': 'running', 'experiments': []}))
    (stage / 'paired-results.json').write_text(json.dumps({'pretend_complete': True}))
    try:
        analysis.stage_gate(stage, 'observations', {})
    except analysis.Pending as error:
        assert error.reason == 'benchmark_stage_not_complete'
    else:
        raise AssertionError('Partial stage with a saved paired report must stay pending')

def reject_identity(callback, message):
    try:
        callback()
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError('Changed or invalid complete scene assets must be rejected')

assets = 'a' * 64
changed_texture_assets = 'b' * 64
assert analysis.checked_scene_assets_sha256({'scene_assets_sha256': assets}) == assets
for digest in (None, '', 'a' * 63, 'A' * 64, 123):
    reject_identity(lambda: analysis.checked_scene_assets_sha256({'scene_assets_sha256': digest}),
                    'valid scene_assets_sha256')
reject_identity(lambda: analysis.checked_scene_assets_sha256(
    {'scene_assets_sha256': changed_texture_assets}, assets), 'differs within the stage')
identity = {'source_versions': {'project': 'fixed'}, 'scene_mesh_sha256': 'c' * 64,
            'scene_assets_sha256': assets}
analysis.verify_stage_source_identity({'observations': identity, 'time': dict(identity)})

# Exercise the actual top-level read-only analysis gate with synthetic audited
# stage records: identical mesh/source, different complete asset (texture) hash.
# No real campaign completion or numerical results are fabricated or written.
different_assets = {'observations': identity,
                    'time': dict(identity, scene_assets_sha256=changed_texture_assets)}
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    target = root / 'must_not_exist.json'
    with patch.object(analysis, 'completed_campaign', return_value=(
            {}, {'smoke': root / 'smoke', 'observations': root / 'observations', 'time': root / 'time'})), \
         patch.object(analysis, 'stage_gate', return_value={}), \
         patch.object(analysis, 'analyzer_module', return_value=object()), \
         patch.object(analysis, 'analyze_stage', side_effect=lambda _, __, stage, *args: different_assets[stage]):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = analysis.main(['--root', str(root), '--output', str(target)])
        rejected = json.loads(output.getvalue())
        assert code == 1 and rejected['status'] == 'rejected'
        assert 'changed source versions or scene assets' in rejected['error']
        assert rejected['quality_summary_generated'] is False
        assert rejected['output_written'] is False and not target.exists()
print('Scratch CPU analysis gates, complete scene asset identity, and arithmetic passed; no quality results generated.')
