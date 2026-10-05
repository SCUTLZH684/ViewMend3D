"""Static environment checks use small files, never render or initialize CUDA."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/activegs/check_environment_cpu.py"
SPEC = importlib.util.spec_from_file_location("environment_cpu_check", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class SceneInputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "scene/habitat").mkdir(parents=True)
        (self.root / "scene/textures").mkdir()
        (self.root / "scene/mesh.ply").write_bytes(b"mesh fixture")
        (self.root / "scene/habitat/stage.json").write_text(json.dumps({"render_asset": "../mesh.ply"}))
        (self.root / "scene/textures/0-color-ptex.hdr").write_bytes(b"texture fixture")
        (self.root / "scene/textures/parameters.json").write_text("{}")

    def tearDown(self):
        self.temporary.cleanup()

    def inspect(self):
        return checker.scene_assets(self.root, "scene/mesh.ply", "scene/habitat/stage.json")

    def test_identity_detects_texture_changes_even_with_identical_mesh(self):
        rows, first = self.inspect()
        self.assertEqual(len(rows), 4)
        (self.root / "scene/textures/0-color-ptex.hdr").write_bytes(b"different texture")
        self.assertNotEqual(first, self.inspect()[1])

    def test_missing_render_or_texture_inputs_fail_before_gpu(self):
        (self.root / "scene/textures/0-color-ptex.hdr").unlink()
        with self.assertRaisesRegex(ValueError, "texture"):
            self.inspect()
        (self.root / "scene/textures/0-color-ptex.hdr").write_bytes(b"texture fixture")
        (self.root / "scene/mesh.ply").unlink()
        with self.assertRaisesRegex(ValueError, "Missing/empty"):
            self.inspect()

    def test_stage_assets_must_stay_in_upstream_checkout(self):
        (self.root / "scene/habitat/stage.json").write_text(json.dumps({"render_asset": "../../../outside.ply"}))
        with self.assertRaises(ValueError):
            self.inspect()

    def test_unexpected_upstream_edits_are_rejected(self):
        outputs = [checker.ACTIVEGS_COMMIT, "mapping/gaussian_map.py"]
        with patch.object(checker.subprocess, "run") as run:
            run.side_effect = [type("Result", (), {"stdout": value})() for value in outputs]
            with self.assertRaisesRegex(ValueError, "Unexpected upstream"):
                checker.source_check(self.root)


if __name__ == "__main__":
    unittest.main()
