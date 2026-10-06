"""Reject a mismatched evaluation reference before publishing scene geometry."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/web'))
from export_reference import reference_source, export_reference


class ReferenceTests(unittest.TestCase):
    def test_source_identity_and_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'mesh.ply'; source.write_bytes(b'fixture-not-real-geometry')
            config = {'scene': {'scene_name': 'replica/office0', 'mesh_path': 'mesh.ply'}}
            (root / 'exp_config.json').write_text(json.dumps(config))
            protocol = {'scene': 'replica/office0', 'scene_mesh_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            self.assertEqual(reference_source(root, protocol, root), (source.resolve(), protocol['scene_mesh_sha256']))
            source.write_bytes(b'different-reference')
            with self.assertRaisesRegex(ValueError, 'source hash'): reference_source(root, protocol, root)
            config['scene']['scene_name'] = 'replica/office1'
            (root / 'exp_config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'office0 scene'): reference_source(root, protocol, root)

    def test_legacy_export_does_not_read_reference(self):
        self.assertIsNone(export_reference(Path('missing'), Path('missing'), {}))


if __name__ == '__main__':
    unittest.main()
