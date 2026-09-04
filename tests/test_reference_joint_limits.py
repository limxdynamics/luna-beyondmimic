import numpy as np
import sys
import unittest
from pathlib import Path

TOOL_DIR = Path(__file__).parents[1] / "tools" / "offline_onnx_exporter"
sys.path.insert(0, str(TOOL_DIR))

import deploy_config  # noqa: E402
from reference_converter import _check_joint_travel  # noqa: E402


class ReferenceJointLimitTest(unittest.TestCase):
    def setUp(self):
        self.positions = np.zeros((3, deploy_config.ACTION_DIM), dtype=np.float64)
        self.column = deploy_config.REFERENCE_JOINT_NAMES.index("left_ankle_roll_joint")
        self.upper = deploy_config.REFERENCE_JOINT_LIMITS[self.column][1]

    def test_small_workspace_miss_is_clipped_to_hard_limit(self):
        self.positions[1, self.column] = self.upper + 0.0125

        clipped = _check_joint_travel(self.positions)

        self.assertEqual(clipped[1, self.column], self.upper)
        self.assertEqual(self.positions[1, self.column], self.upper + 0.0125)

    def test_large_workspace_miss_is_rejected(self):
        self.positions[1, self.column] = self.upper + deploy_config.JOINT_LIMIT_TOLERANCE + 0.001

        with self.assertRaisesRegex(ValueError, "by more than"):
            _check_joint_travel(self.positions)

    def test_in_range_reference_is_unchanged(self):
        clipped = _check_joint_travel(self.positions)

        np.testing.assert_array_equal(clipped, self.positions)


if __name__ == "__main__":
    unittest.main()
