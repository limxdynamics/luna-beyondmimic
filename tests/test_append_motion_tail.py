import numpy as np
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from append_motion_tail import check_final_frame_is_static  # noqa: E402


def motion_with_speeds(*, joint: float = 0.0, linear: float = 0.0, angular: float = 0.0) -> dict:
    return {
        "joint_vel": np.array([[joint]], dtype=np.float32),
        "body_lin_vel_w": np.array([[[linear, 0.0, 0.0]]], dtype=np.float32),
        "body_ang_vel_w": np.array([[[angular, 0.0, 0.0]]], dtype=np.float32),
    }


class FinalFrameStaticCheckTest(unittest.TestCase):
    def test_accepts_small_final_body_rotation(self):
        check_final_frame_is_static(motion_with_speeds(angular=0.15), force=False)

    def test_rejects_body_rotation_above_separate_angular_limit(self):
        with self.assertRaises(SystemExit):
            check_final_frame_is_static(motion_with_speeds(angular=0.21), force=False)

    def test_joint_and_linear_limits_remain_strict(self):
        with self.assertRaises(SystemExit):
            check_final_frame_is_static(motion_with_speeds(joint=0.11), force=False)
        with self.assertRaises(SystemExit):
            check_final_frame_is_static(motion_with_speeds(linear=0.11), force=False)


if __name__ == "__main__":
    unittest.main()
