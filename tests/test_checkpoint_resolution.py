import sys
import tempfile
import unittest
from pathlib import Path

RSL_RL_SCRIPTS = Path(__file__).parents[1] / "scripts" / "rsl_rl"
sys.path.insert(0, str(RSL_RL_SCRIPTS))

from cli_args import resolve_direct_checkpoint  # noqa: E402


class DirectCheckpointResolutionTest(unittest.TestCase):
    def test_existing_relative_path_is_resolved_directly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model_14999.pt"
            checkpoint.write_bytes(b"checkpoint")

            resolved = resolve_direct_checkpoint(str(checkpoint))

            self.assertEqual(resolved, str(checkpoint.resolve()))

    def test_bare_filename_regex_keeps_legacy_log_lookup(self):
        self.assertIsNone(resolve_direct_checkpoint(r"model_14999\.pt"))
        self.assertIsNone(resolve_direct_checkpoint(r"model_.*\.pt"))

    def test_missing_path_like_argument_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            resolve_direct_checkpoint("amos/input/missing.pt")

    def test_none_does_not_select_direct_checkpoint(self):
        self.assertIsNone(resolve_direct_checkpoint(None))


if __name__ == "__main__":
    unittest.main()
