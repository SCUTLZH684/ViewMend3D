"""Sensor-record/export integrity and constrained launch checks; CPU fixtures only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/web"))
from viewmend3d.replay import record_observation
from export_replay import export_replay
import server


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name) / "run"
        self.experiment = self.run / "experiments/benchmark/replica/office0/defect_guarded/0"
        self.output = Path(self.temp.name) / "output"
        (self.experiment / "diagnostics").mkdir(parents=True)
        (self.run / "common-prefix/seed_0/diagnostics").mkdir(parents=True)
        self.output.mkdir()
        self.manifest = {"method_id": "defect_guarded", "camera_convention": "OpenCV camera-to-world",
            "protocol": {"seed": 0, "replay_recording": {"extra_recording_io": True},
                         "protocol": {"observations": 8, "prefix": 1, "checkpoint_every": 1}},
            "cameras": [], "checkpoints": []}
        for event in range(1, 9):
            pose = np.eye(4, dtype=np.float32); pose[0, 3] = event
            frame = {"rgb": np.full((3, 2, 3), .5), "depth": np.asarray([[0,1,2],[3,4,np.nan]],dtype=np.float32),
                     "extrinsic": pose, "intrinsic": np.eye(3), "depth_range": np.asarray([0,5])}
            record_observation(frame, self.experiment, event)
            self.manifest["cameras"].append({"position": pose[:3,3].tolist(),"rotation":pose[:3,:3].reshape(-1).tolist()})
            self.manifest["checkpoints"].append({"observation_count":event,"update_event":event})
            decision = {"step":event, "future_candidate_observation_queries":0, "selected_pose":pose.tolist()}
            if event == 1:
                decision.update(initialization=True,candidate_poses=[],selected_index=None)
                path = self.run / "common-prefix/seed_0/diagnostics" / f"step_{event:04}.json"
            else:
                other = pose.copy(); other[1,3]=2
                decision.update(candidate_poses=[pose.tolist(),other.tolist()],selected_index=0,baseline_selected_index=1,
                    reachable=[True,True],final_scores=[.8,.7],exploration=[.2,.3],uncertainty=[.4,.3],defect=[.8,.1],
                    path_lengths=[1,2],baseline_scores=[.6,.7],geometry_bonus=[.2,0])
                path = self.experiment / "diagnostics" / f"step_{event:04}.json"
            path.write_text(json.dumps(decision))

    def tearDown(self):
        self.temp.cleanup()

    def export(self):
        return export_replay(self.run, self.experiment, self.output, self.manifest)

    def test_actual_sequence_and_raw_depth(self):
        descriptor = self.export()
        replay = json.loads((self.output / descriptor["file"]).read_text())
        self.assertEqual([f["event"] for f in replay["frames"]], list(range(1,9)))
        self.assertTrue(replay["frames"][0]["decision"]["initialization"])
        self.assertEqual(replay["frames"][1]["decision"]["selected_index"],0)
        self.assertEqual(replay["frames"][1]["pose"][0][3],2)
        self.assertEqual(replay["frames"][0]["valid_depth_pixels"],4)
        depth = np.asarray(Image.open(self.output / replay["frames"][0]["depth"]))
        np.testing.assert_array_equal(depth[0,0],[52,57,65])
        self.assertFalse(list(self.output.rglob('*.npy')))

    def test_tampering_rejected(self):
        path = self.experiment / "observations/frame_0002_rgb.png"
        path.write_bytes(path.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): self.export()

    def test_decision_pose_mismatch_rejected(self):
        path=self.experiment / "diagnostics/step_0002.json"
        value=json.loads(path.read_text()); value['selected_pose'][0][3]=999
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'selected pose'): self.export()

    def test_candidate_sensor_query_rejected(self):
        path=self.experiment / "diagnostics/step_0002.json"
        value=json.loads(path.read_text()); value['future_candidate_observation_queries']=1
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'future sensor'): self.export()

    def test_missing_frame_and_camera_mismatch_rejected(self):
        manifest=copy.deepcopy(self.manifest)
        self.manifest['cameras'][1]['position'][0]=999
        with self.assertRaisesRegex(ValueError,'saved acquisition'): self.export()
        self.manifest=manifest
        (self.experiment / 'observations/frame_0008.json').unlink()
        with self.assertRaisesRegex(ValueError,'incomplete'): self.export()

    def test_recording_never_overwrites(self):
        record=json.loads((self.experiment/'observations/frame_0001.json').read_text())
        frame={'rgb':np.zeros((3,2,3)), 'depth':np.ones((2,3)), 'extrinsic':np.eye(4), 'intrinsic':np.eye(3), 'depth_range':[0,5]}
        with self.assertRaisesRegex(ValueError,'overwritten'): record_observation(frame,self.experiment,1)
        self.assertEqual(json.loads((self.experiment/'observations/frame_0001.json').read_text()),record)

    def test_legacy_has_no_synthetic_replay(self):
        self.manifest['protocol'].pop('replay_recording')
        with self.assertRaisesRegex(ValueError,'explicit replay'): self.export()
        self.assertIsNone(export_replay(self.run,self.run/'unrecorded',self.output,self.manifest))

    def test_launcher_is_separate_and_idle_only(self):
        idle=[{'index':1,'available':True}]
        payload={'gpu':1,'protocol':'replay','method':'defect_guarded','seed':0,'frames':8}
        job=server.parse_job(payload,idle)
        env={'ACTIVEGS_PYTHON':'python','ACTIVEGS_ROOT':'upstream','RUN_DIR':'run','BENCHMARK_RECIPE':job['recipe']}
        command=server.benchmark_command(ROOT,job,env)
        self.assertEqual(command[command.index('--frames')+1],'8')
        self.assertEqual(command[command.index('--prefix-frames')+1],'1')
        self.assertIn('--record-replay',command)
        for key,value in [('frames',60),('method','confidence_nooracle'),('seed',True),('gpu',True)]:
            with self.subTest(key=key), self.assertRaises(ValueError): server.parse_job({**payload,key:value},idle)
        with self.assertRaises(RuntimeError): server.parse_job(payload,[{'index':1,'available':False}])
        old=server.parse_job({**payload,'protocol':'observations','frames':60},idle)
        oldcommand=server.benchmark_command(ROOT,old,env)
        self.assertNotIn('--record-replay',oldcommand)
        self.assertEqual(oldcommand[oldcommand.index('--frames')+1],'60')
        self.assertEqual(oldcommand[oldcommand.index('--prefix-frames')+1],'20')


if __name__ == '__main__':
    unittest.main()
