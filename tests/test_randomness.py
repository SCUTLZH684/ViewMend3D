"""Run with CUDA_VISIBLE_DEVICES='' to test RNG isolation entirely on CPU."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch environment is required")
class RandomDomainChecks(unittest.TestCase):
    def test_sensor_and_mapping_unaffected_by_planner_random_work(self):
        import numpy as np
        import torch
        from viewmend3d.randomness import random_domain
        def sample(extra_planning):
            with random_domain(2, "planning", 21):
                np.random.normal(size=extra_planning)
                torch.rand(extra_planning)
                random.random()
            with random_domain(2, "sensor", 21):
                sensor = np.random.normal(size=16)
            with random_domain(2, "mapping", 21):
                mapping = torch.rand(16)
            return sensor, mapping
        first, second = sample(1), sample(1000)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_context_restores_outer_rngs_even_when_operation_fails(self):
        import numpy as np
        import torch
        from viewmend3d.randomness import random_domain
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with random_domain(0, "mapping", 20):
                random.random()
                np.random.uniform(size=20)
                torch.rand(20)
                raise RuntimeError("fixture")
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(np.random.get_state()[0], numpy_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))


if __name__ == "__main__":
    unittest.main()
