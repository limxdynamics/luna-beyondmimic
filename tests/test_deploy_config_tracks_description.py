"""Pin deploy_config's numbers to the description they were read from.

Every number in ``deploy_config`` that describes the robot rather than the policy comes from the
``luna-description`` submodule. Nothing stops the two from drifting apart when the submodule is
bumped, and a stale limit or gain is invisible until it reaches hardware. These tests recompute
them from the description and fail on any disagreement.

The submodule is optional - a checkout that has not initialised it, including CI, skips these.
"""

import importlib
import sys
import unittest
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

TOOL_DIR = Path(__file__).parents[1] / "tools" / "offline_onnx_exporter"
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import deploy_config  # noqa: E402

npy_to_npz = importlib.import_module("npy_to_npz")

DESCRIPTION = Path(__file__).parents[1] / "robot_descriptions/luna-description/HU_L04_description"
URDF = DESCRIPTION / "urdf/HU_L04_01.urdf"
MJCF = DESCRIPTION / "xml/HU_L04_01.xml"
USD = DESCRIPTION / "usd_parallel/HU_L04_01.usd"

# The six reference columns that carry an achilles motor rather than the joint they are named
# after. Only the MJCF declares a range or an armature for those motors.
MOTOR_FOR_JOINT = {
    "left_ankle_pitch_joint": "left_A_achilles_joint",
    "left_ankle_roll_joint": "left_B_achilles_joint",
    "right_ankle_pitch_joint": "right_A_achilles_joint",
    "right_ankle_roll_joint": "right_B_achilles_joint",
    "waist_roll_joint": "waist_A_joint",
    "waist_pitch_joint": "waist_B_joint",
}

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def urdf_limits():
    return {
        joint.get("name"): (float(joint.find("limit").get("lower")), float(joint.find("limit").get("upper")))
        for joint in ET.parse(URDF).getroot().iter("joint")
        if joint.find("limit") is not None
    }


def urdf_efforts():
    return {
        joint.get("name"): float(joint.find("limit").get("effort"))
        for joint in ET.parse(URDF).getroot().iter("joint")
        if joint.find("limit") is not None
    }


def mjcf_attribute(name):
    values = {}
    for joint in ET.parse(MJCF).getroot().iter("joint"):
        if joint.get("name") and joint.get(name):
            values[joint.get("name")] = joint.get(name)
    return values


@unittest.skipUnless(URDF.exists() and MJCF.exists(), "luna-description submodule is not checked out")
class DeployConfigTracksDescriptionTest(unittest.TestCase):
    def test_reference_joint_limits_match_the_description(self):
        limits = urdf_limits()
        ranges = mjcf_attribute("range")

        for column, name in enumerate(deploy_config.REFERENCE_JOINT_NAMES):
            with self.subTest(joint=name):
                if name in MOTOR_FOR_JOINT:
                    lower, upper = (float(value) for value in ranges[MOTOR_FOR_JOINT[name]].split())
                else:
                    lower, upper = limits[name]
                self.assertAlmostEqual(deploy_config.REFERENCE_JOINT_LIMITS[column][0], lower, places=6)
                self.assertAlmostEqual(deploy_config.REFERENCE_JOINT_LIMITS[column][1], upper, places=6)

    def test_waist_motor_travel_is_the_joint_travel_reflected(self):
        """The one place where reading the URDF instead of the MJCF would silently invert a limit."""
        ranges = mjcf_attribute("range")
        joint_lower, joint_upper = urdf_limits()["waist_pitch_joint"]
        motor_lower, motor_upper = (float(value) for value in ranges["waist_B_joint"].split())

        self.assertAlmostEqual(motor_lower, -joint_upper, places=6)
        self.assertAlmostEqual(motor_upper, -joint_lower, places=6)

    def test_gains_are_derived_from_the_description_armature(self):
        armature = {name: float(value) for name, value in mjcf_attribute("armature").items()}
        efforts = urdf_efforts()

        for index, name in enumerate(deploy_config.DEPLOY_JOINT_NAMES):
            with self.subTest(joint=name):
                inertia = armature[MOTOR_FOR_JOINT.get(name, name)]
                stiffness = inertia * NATURAL_FREQ**2
                damping = 2.0 * DAMPING_RATIO * inertia * NATURAL_FREQ

                self.assertAlmostEqual(deploy_config.DEPLOY_JOINT_STIFFNESS[index], stiffness, places=3)
                self.assertAlmostEqual(deploy_config.DEPLOY_JOINT_DAMPING[index], damping, places=3)
                self.assertAlmostEqual(
                    deploy_config.DEPLOY_ACTION_SCALE[index], 0.25 * efforts[name] / stiffness, places=3
                )


class BodyOrderIsConsistentTest(unittest.TestCase):
    """The npz body axis is positional, so a wrong length or a wrong slot is a silent corruption.

    ``deploy_config`` cannot import the converter - it has to run standalone outside the
    repository - so the list exists twice. This is the check that keeps the copies honest.
    """

    def test_deploy_config_matches_the_converter(self):
        self.assertEqual(
            list(deploy_config.TRAINING_BODY_NAMES),
            list(npy_to_npz.HU_L04_PARALLEL_BODY_ORDER),
        )


def usd_body_order(path):
    """Bodies in articulation order: breadth-first from the root over the joint graph.

    This reproduces the order Isaac Lab reports in ``robot.body_names``, which is what the npz
    body axis has to line up with.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    children = defaultdict(list)
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if parent and child:
                children[parent[0].name].append(child[0].name)

    order, seen, queue = [], {"base_link"}, deque(["base_link"])
    while queue:
        body = queue.popleft()
        order.append(body)
        for child in children[body]:
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return order


@unittest.skipUnless(USD.exists(), "luna-description submodule is not checked out")
class BodyOrderTracksDescriptionTest(unittest.TestCase):
    def setUp(self):
        try:
            import pxr  # noqa: F401
        except ImportError:
            self.skipTest("pxr (usd-core) is not installed")

    def test_converter_body_order_matches_the_usd(self):
        """HU_L04 added the fixed-joint base_imu that HU_L03 did not have.

        It shifts every body after it by one slot, which the length check in verify_motion_npz
        catches, and would otherwise mismatch every body silently if the count happened to agree.
        """
        self.assertEqual(list(npy_to_npz.HU_L04_PARALLEL_BODY_ORDER), usd_body_order(USD))


if __name__ == "__main__":
    unittest.main()
