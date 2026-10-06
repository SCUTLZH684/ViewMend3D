"""Package only referenced display assets, never training data or checkpoints.

Standard-library tool. Share the archive separately from Git; extract into a
fresh clone and use server.py --read-only without CUDA/Habitat/ML dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            value.update(part)
    return value.hexdigest()


def referenced_assets(folder):
    folder = folder.resolve()
    files = {}

    def include(name, expected=None):
        if not isinstance(name, str) or '\\' in name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Display references must stay inside the asset folder')
        path = (folder / name).resolve()
        path.relative_to(folder)
        if path.suffix not in ('.json', '.png', '.ply') or not path.is_file():
            raise ValueError(f'Missing or unsupported display asset: {name}')
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Display hash mismatch: {name}')
        files[name] = {'sha256': actual, 'bytes': path.stat().st_size}
        return path

    manifest = json.loads(include('manifest.json').read_text(encoding='utf-8'))
    if manifest.get('id') != folder.name or manifest.get('status') != 'completed' or not manifest.get('replay') or not manifest.get('ground_truth'):
        raise ValueError('Require a completed recorded replay with Ground Truth')
    include('summary.json')
    replay = json.loads(include(manifest['replay']['file'], manifest['replay']['sha256']).read_text(encoding='utf-8'))
    frames = replay['frames']
    if (len(frames) != 8 or replay.get('recorded_sensor_only') is not True
            or len(manifest['checkpoints']) != 8 or [frame['event'] for frame in frames] != list(range(1,9))):
        raise ValueError('Require all eight actual acquisitions and maps')
    reference = manifest['ground_truth']
    if reference['source_sha256'] != manifest['protocol']['scene_mesh_sha256']:
        raise ValueError('Ground Truth does not match evaluation source')
    include(reference['mesh'], reference['preview_sha256'])
    if manifest.get('initial_rgb'):
        include(manifest['initial_rgb'])
    for checkpoint in manifest['checkpoints']:
        include(checkpoint['mesh'])
        heatmap = (checkpoint.get('diagnostic') or {}).get('heatmap')
        if heatmap:
            include(heatmap)
    for frame in frames:
        for key in ('rgb', 'depth'):
            include(frame[key], frame['sha256'][key])
    return {'schema': 'viewmend-display-package-v1', 'id': folder.name,
            'scope': 'existing_eight_frame_replay_not_new_reconstruction', 'files': files,
            'replay_sha256': manifest['replay']['sha256'], 'ground_truth_source_sha256': reference['source_sha256']}


def pack(folder, destination):
    if destination.exists():
        raise ValueError('Package destination already exists; refusing to overwrite')
    index = referenced_assets(folder)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"runs/web-assets/{index['id']}/"
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(index['files']):
            archive.write(folder / name, prefix + name)
        archive.writestr(prefix + 'package-index.json', json.dumps(index, ensure_ascii=False, indent=2))
    return {'archive': destination.name, 'sha256': digest(destination), 'bytes': destination.stat().st_size,
            'file_count': len(index['files']), 'id': index['id']}


def verify(folder):
    index = json.loads((folder / 'package-index.json').read_text(encoding='utf-8'))
    actual = referenced_assets(folder)
    if index != actual:
        raise ValueError('Package contents differ from the recorded checksums')
    return {'id': index['id'], 'verified_files': len(index['files']), 'all_match': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assets', type=Path, help='Completed eight-frame display asset folder')
    parser.add_argument('archive', type=Path, nargs='?', help='New ZIP path; extract into a fresh clone')
    parser.add_argument('--verify', action='store_true', help='Verify an extracted package before replay')
    args = parser.parse_args()
    if args.verify and args.archive or not args.verify and args.archive is None:
        parser.error('Use ASSETS ARCHIVE.zip to package, or ASSETS --verify to check')
    result = verify(args.assets) if args.verify else pack(args.assets, args.archive)
    print(json.dumps(result, ensure_ascii=False))
