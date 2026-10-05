"""Synthetic file-chain fixtures; these are not reconstruction quality results."""
import json
import pickle
import struct
import sys
import tempfile
import unittest
from pathlib import Path
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/activegs"))
from check_artifacts import check, ply_counts

CAMERA = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
          1, 0, .5, 0, 1, .5, 0, 0, 1]
PLY_HEADER = ("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n"
              "property float y\nproperty float z\nelement face 1\n"
              "property list uchar int vertex_indices\nend_header\n")
PLY_BODY = "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"


def write_output_chain(path, result):
    """Write tiny valid file containers, without generating a GPU result."""
    folder = path / "map"
    folder.mkdir(exist_ok=True)
    (path / "final_result.json").write_text(json.dumps(result), encoding="utf-8")
    (folder / "record_info.txt").write_text("".join(
        f"{event} {seconds} {length}\n" for event, seconds, length in
        zip(result["step"], result["time"], result["path_length"])), encoding="utf-8")
    counts = result.get("observation_count", result["step"])
    for event, count in zip(result["step"], counts):
        with zipfile.ZipFile(folder / f"map_{event:03}.th", "w") as archive:
            archive.writestr("fixture/data.pkl", pickle.dumps({"fixture_only": True}))
            archive.writestr("fixture/data/0", struct.pack("<f", 0))
        (folder / f"cameras_{event:03}.pkl").write_bytes(pickle.dumps([list(CAMERA) for _ in range(count)]))
        (folder / f"mesh_{event:03}.ply").write_text(PLY_HEADER + PLY_BODY, encoding="ascii")


class ArtifactChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.result = {"step": [2, 4], "time": [1, 2], "path_length": [0, 1],
                       "mesh_accuracy": [1, 1], "mesh_completion": [2, 2],
                       "mesh_completion_ratio": [80, 90], "mesh_chamfer_distance": [.015, .015]}
        write_output_chain(self.path, self.result)

    def tearDown(self):
        self.temp.cleanup()

    def test_plain_original_counts_and_current_fingerprints(self):
        report = check(self.path)
        self.assertTrue(report["artifact_chain_passed"])
        self.assertEqual(report["observation_count"], [2, 4])
        self.assertEqual(len(report["files"]), 8)
        self.assertEqual(report["validation_version"], "stdlib-current-artifacts-v2")

    def test_record_and_metric_time_or_path_must_agree(self):
        for field in ("time", "path_length"):
            with self.subTest(field=field):
                changed = {**self.result, field: [0, 3]}
                (self.path / "final_result.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "record and result"):
                    check(self.path)
        (self.path / "final_result.json").write_text(json.dumps(self.result), encoding="utf-8")

    def test_rejects_nonfinite_decreasing_or_fractional_records(self):
        for records in ("2 nan 0\n4 2 1\n", "2 2 1\n4 1 1\n", "2.5 1 0\n4 2 1\n", "4 1 0\n2 2 1\n"):
            with self.subTest(records=records):
                (self.path / "map/record_info.txt").write_text(records, encoding="utf-8")
                with self.assertRaises(ValueError):
                    check(self.path)

    def test_camera_count_and_prefix_are_actual_file_checks(self):
        self.result.update(observation_count=[2, 3], update_event=[2, 4])
        (self.path / "final_result.json").write_text(json.dumps(self.result), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "camera count"):
            check(self.path)
        self.result["observation_count"] = [2, 4]
        (self.path / "final_result.json").write_text(json.dumps(self.result), encoding="utf-8")
        cameras = [list(CAMERA) for _ in range(4)]
        cameras[0][3] = .1
        (self.path / "map/cameras_004.pkl").write_bytes(pickle.dumps(cameras))
        with self.assertRaisesRegex(ValueError, "acquisition prefix"):
            check(self.path)

    def test_rejects_camera_geometry_and_nonfinite_values(self):
        for column, invalid in ((0, 2), (15, 0), (16, 0), (20, float("nan"))):
            with self.subTest(column=column):
                cameras = [list(CAMERA) for _ in range(2)]
                cameras[0][column] = invalid
                (self.path / "map/cameras_002.pkl").write_bytes(pickle.dumps(cameras))
                with self.assertRaises(ValueError):
                    check(self.path)

    def test_rejects_header_only_nonfinite_and_invalid_face_meshes(self):
        mesh = self.path / "map/mesh_004.ply"
        for body in ("", PLY_BODY.replace("0 1 0", "nan 1 0"), PLY_BODY.replace("3 0 1 2", "3 0 1 3")):
            with self.subTest(body=body):
                mesh.write_text(PLY_HEADER + body, encoding="ascii")
                with self.assertRaises(ValueError):
                    check(self.path)

    def test_binary_ply_payload_validation(self):
        mesh = self.path / "binary.ply"
        for encoding, endian in (("binary_little_endian", "<"), ("binary_big_endian", ">")):
            header = PLY_HEADER.replace("format ascii", f"format {encoding}").encode("ascii")
            body = struct.pack(endian + "9fB3i", 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0, 1, 2)
            mesh.write_bytes(header + body)
            self.assertEqual(ply_counts(mesh), {"vertex": 3, "face": 1})
            mesh.write_bytes(header + body[:-1])
            with self.assertRaisesRegex(ValueError, "Truncated"):
                ply_counts(mesh)

    def test_rejects_truncated_map_container(self):
        map_file = self.path / "map/map_004.th"
        map_file.write_bytes(map_file.read_bytes()[:20])
        with self.assertRaisesRegex(ValueError, "Torch ZIP"):
            check(self.path)


if __name__ == "__main__":
    unittest.main()
