"""Display-only preview of the exact reference mesh used by a recorded run.

Never imported by acquisition/planning. The full source stays in the dataset;
only a reduced, identity-checked mesh is published under ignored web assets.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
        return result.hexdigest()


def reference_source(experiment, protocol, upstream):
    config = json.loads((experiment / 'exp_config.json').read_text(encoding='utf-8'))
    scene = config['scene']
    if scene['scene_name'] != 'replica/office0' or protocol.get('scene') not in ('office0', 'replica/office0'):
        raise ValueError('Reference preview requires the recorded Replica office0 scene')
    path = Path(scene['mesh_path'])
    source = (path if path.is_absolute() else upstream / path).resolve()
    expected = protocol.get('scene_mesh_sha256')
    if not isinstance(expected, str) or len(expected) != 64 or digest(source) != expected:
        raise ValueError('Ground Truth source hash differs from the recorded evaluation reference')
    return source, expected


def export_reference(experiment, output, manifest, upstream=None, faces=80000):
    if not manifest.get('replay'):
        return None
    if faces < 1000:
        raise ValueError('Reference preview needs at least 1000 faces')
    source, source_hash = reference_source(experiment, manifest['protocol'], upstream or ROOT / 'external/active-gs')
    import open3d as o3d
    import numpy as np

    mesh = o3d.io.read_triangle_mesh(str(source))
    original = {'vertices': len(mesh.vertices), 'faces': len(mesh.triangles)}
    if not original['faces'] or not mesh.has_vertex_colors():
        raise ValueError('Ground Truth needs nonempty geometry and actual vertex colors')
    bounds = [mesh.get_min_bound().tolist(), mesh.get_max_bound().tolist()]
    if not np.isfinite(bounds).all():
        raise ValueError('Ground Truth has nonfinite coordinates')
    if original['faces'] > faces:
        mesh = mesh.simplify_quadric_decimation(faces)
    # No transform/recentering: reference and recorded OpenCV poses share world coordinates.
    filename = f'ground_truth_{source_hash[:12]}_{faces}.ply'
    output.mkdir(parents=True, exist_ok=True)
    target = output / filename
    temporary = output / f'.{filename}'
    if not o3d.io.write_triangle_mesh(str(temporary), mesh):
        raise ValueError('Failed to write Ground Truth preview')
    if target.exists() and digest(target) != digest(temporary):
        temporary.unlink()
        raise ValueError('Existing Ground Truth preview differs; refusing to overwrite')
    temporary.replace(target)
    return {'schema': 'viewmend-reference-preview-v1', 'scene': 'Replica office0',
            'mesh': filename, 'source_sha256': source_hash, 'preview_sha256': digest(target),
            'bounds': bounds, 'up_axis': 'z', 'transform': np.eye(4).tolist(),
            'camera_convention': manifest['camera_convention'], 'original': original,
            'preview': {'vertices': len(mesh.vertices), 'faces': len(mesh.triangles)},
            'usage': 'display_and_evaluation_only_not_planner_input'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attach verified Ground Truth to an existing completed replay; preserve all observations and results')
    parser.add_argument('experiment', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external/active-gs')
    args = parser.parse_args()
    path = args.output / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    reference = export_reference(args.experiment, args.output, manifest, args.upstream)
    if reference is None:
        parser.error('Selected experiment has no recorded replay')
    manifest['ground_truth'] = reference
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)
    print(json.dumps(reference, ensure_ascii=False))
