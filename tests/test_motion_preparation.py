import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RSL_RL_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl"
if str(RSL_RL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPTS))

import motion_preparation


class MotionPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.motion = self.root / "motion.npy"
        self.motion.write_bytes(b"trusted test placeholder")
        self.parser = argparse.ArgumentParser()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prepares_npy_and_replaces_argument_with_temporary_npz(self):
        args = argparse.Namespace(
            motion_file=str(self.motion),
            motion_tail_seconds=7.5,
            motion_max_unreachable_error=0.2,
            motion_force_tail=True,
        )
        commands = []

        def run(command):
            commands.append(command)
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"prepared")
            return mock.Mock(returncode=0)

        with mock.patch.object(motion_preparation.subprocess, "run", side_effect=run):
            scratch = motion_preparation.prepare_motion_file(args, self.parser)

        prepared = Path(args.motion_file)
        self.assertTrue(prepared.is_file())
        self.assertEqual(prepared.name, "motion.npz")
        self.assertIn(str(self.motion), commands[0])
        self.assertIn("7.5", commands[0])
        self.assertIn("0.2", commands[0])
        self.assertIn("--force_tail", commands[0])

        scratch.cleanup()
        self.assertFalse(prepared.exists())

    def test_rejects_npz_as_a_user_input(self):
        args = argparse.Namespace(
            motion_file=str(self.root / "motion.npz"),
            motion_tail_seconds=10.0,
            motion_max_unreachable_error=None,
            motion_force_tail=False,
        )

        with self.assertRaises(SystemExit) as raised:
            motion_preparation.prepare_motion_file(args, self.parser)

        self.assertEqual(raised.exception.code, 2)

    def test_rejects_a_missing_preparation_output(self):
        args = argparse.Namespace(
            motion_file=str(self.motion),
            motion_tail_seconds=10.0,
            motion_max_unreachable_error=None,
            motion_force_tail=False,
        )

        with (
            mock.patch.object(
                motion_preparation.subprocess,
                "run",
                return_value=mock.Mock(returncode=0),
            ),
            self.assertRaisesRegex(SystemExit, "without writing"),
        ):
            motion_preparation.prepare_motion_file(args, self.parser)


if __name__ == "__main__":
    unittest.main()
