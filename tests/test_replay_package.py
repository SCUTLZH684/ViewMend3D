"""Portable display package integrity; synthetic CPU fixtures, no model runs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/web'))
from pack_replay import digest, pack, verify


class ReplayPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.assets = self.root / 'fixture-demo'; self.assets.mkdir()
        (self.assets / 'replay').mkdir()
        (self.assets / 'reference.ply').write_bytes(b'fixture reference preview')
        frames = []
        checkpoints = []
        for event in range(1,9):
            hashes = {}
            files = {}
            for key in ('rgb', 'depth'):
                name = f'replay/frame_{event}_{key}.png'
                (self.assets / name).write_bytes(f'fixture {event} {key}'.encode())
                files[key] = name; hashes[key] = digest(self.assets / name)
            frames.append({'event': event, **files, 'sha256': hashes})
            name = f'mesh_{event}.ply'; (self.assets / name).write_bytes(b'fixture map')
            checkpoints.append({'mesh': name})
        (self.assets / 'replay.json').write_text(json.dumps({'recorded_sensor_only': True, 'frames': frames}))
        self.manifest = {'id': self.assets.name, 'status': 'completed', 'checkpoints': checkpoints,
            'replay': {'file': 'replay.json', 'sha256': digest(self.assets / 'replay.json')},
            'ground_truth': {'mesh': 'reference.ply', 'source_sha256': 'a'*64, 'preview_sha256': digest(self.assets / 'reference.ply')},
            'protocol': {'scene_mesh_sha256': 'a'*64}}
        self.save_manifest()
        (self.assets / 'summary.json').write_text('{}')
        (self.assets / 'unused-old-reference.ply').write_bytes(b'unused')
        (self.assets / 'private-training.npy').write_bytes(b'not for display')
        self.archive = self.root / 'demo.zip'

    def save_manifest(self):
        (self.assets / 'manifest.json').write_text(json.dumps(self.manifest))

    def tearDown(self): self.temp.cleanup()

    def test_roundtrip_same_bytes_excludes_unreferenced_and_training_files(self):
        result = pack(self.assets, self.archive)
        target = self.root / 'clone'
        with zipfile.ZipFile(self.archive) as archive:
            self.assertFalse(any('unused-old' in name or name.endswith('.npy') for name in archive.namelist()))
            archive.extractall(target)
        extracted = target / 'runs/web-assets' / self.assets.name
        self.assertTrue(verify(extracted)['all_match'])
        self.assertEqual((extracted / 'manifest.json').read_bytes(), (self.assets / 'manifest.json').read_bytes())
        original_hash = digest(self.archive)
        with self.assertRaisesRegex(ValueError, 'already exists'): pack(self.assets, self.archive)
        self.assertEqual(digest(self.archive), original_hash)
        (extracted / 'mesh_2.ply').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksums'): verify(extracted)
        self.assertGreater(result['file_count'], 20)

    def test_tampered_sensor_rejected_before_archive_is_created(self):
        (self.assets / 'replay/frame_2_rgb.png').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): pack(self.assets, self.archive)
        self.assertFalse(self.archive.exists())

    def test_escaping_reference_rejected(self):
        self.manifest['checkpoints'][1]['mesh'] = '../outside.ply'; self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'stay inside'): pack(self.assets, self.archive)


if __name__ == '__main__': unittest.main()
