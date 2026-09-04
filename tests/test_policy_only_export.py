import numpy as np
import os
import subprocess
import sys
import tempfile
import torch
import unittest
from pathlib import Path

import onnx
from onnx.reference import ReferenceEvaluator

TOOL_DIR = Path(__file__).parents[1] / "tools" / "offline_onnx_exporter"
sys.path.insert(0, str(TOOL_DIR))

import deploy_config  # noqa: E402
from policy_exporter import build_export_module, export_policy  # noqa: E402


class PolicyOnlyExportTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.checkpoint = self.root / "model.pt"
        self.output = self.root / "policy.onnx"
        self._write_checkpoint()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _checkpoint(self):
        hidden = 4
        first_weight = torch.arange(
            hidden * deploy_config.OBSERVATION_DIM,
            dtype=torch.float32,
        ).reshape(hidden, deploy_config.OBSERVATION_DIM)
        first_weight /= 1000.0
        second_weight = torch.arange(
            deploy_config.ACTION_DIM * hidden,
            dtype=torch.float32,
        ).reshape(deploy_config.ACTION_DIM, hidden)
        second_weight /= 100.0
        return {
            "model_state_dict": {
                "std": torch.ones(deploy_config.ACTION_DIM),
                "actor.0.weight": first_weight,
                "actor.0.bias": torch.tensor([0.1, -0.2, 0.3, -0.4]),
                "actor.2.weight": second_weight,
                "actor.2.bias": torch.arange(deploy_config.ACTION_DIM, dtype=torch.float32) / 10.0,
            },
            "obs_norm_state_dict": {
                "_mean": torch.linspace(
                    -1.0,
                    1.0,
                    deploy_config.OBSERVATION_DIM,
                ).reshape(1, -1),
                "_std": torch.linspace(
                    1.0,
                    2.0,
                    deploy_config.OBSERVATION_DIM,
                ).reshape(1, -1),
            },
        }

    def _write_checkpoint(self, checkpoint=None, path=None):
        torch.save(checkpoint or self._checkpoint(), path or self.checkpoint)

    def test_fixed_deployment_contract(self):
        self.assertEqual(deploy_config.OBSERVATION_DIM, 141)
        self.assertEqual(deploy_config.ACTION_DIM, 27)
        self.assertEqual(deploy_config.observation_width(), 141)
        self.assertEqual(
            deploy_config.OBSERVATION_NAMES,
            ("command", "base_ang_vel", "joint_pos", "joint_vel", "actions", "projected_gravity"),
        )

    def test_build_module_uses_actor_and_normalizer(self):
        module = build_export_module(self.checkpoint)
        observations = torch.linspace(-2.0, 2.0, deploy_config.OBSERVATION_DIM).reshape(1, -1)

        with torch.no_grad():
            actual = module(observations)

        checkpoint = self._checkpoint()
        normalized = (observations - checkpoint["obs_norm_state_dict"]["_mean"]) / (
            checkpoint["obs_norm_state_dict"]["_std"] + deploy_config.NORMALIZER_EPSILON
        )
        hidden = torch.nn.functional.elu(
            normalized @ checkpoint["model_state_dict"]["actor.0.weight"].T
            + checkpoint["model_state_dict"]["actor.0.bias"]
        )
        expected = (
            hidden @ checkpoint["model_state_dict"]["actor.2.weight"].T + checkpoint["model_state_dict"]["actor.2.bias"]
        )
        self.assertEqual(tuple(actual.shape), (1, deploy_config.ACTION_DIM))
        torch.testing.assert_close(actual, expected)

    def test_exported_onnx_matches_checkpoint_and_metadata(self):
        run_path = "TASK_TEST/model_14999.pt"
        export_policy(self.checkpoint, self.output, run_path=run_path)

        model = onnx.load(self.output)
        onnx.checker.check_model(model)
        self.assertEqual([value.name for value in model.graph.input], ["obs"])
        self.assertEqual([value.name for value in model.graph.output], ["actions"])
        self.assertEqual(
            [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim],
            [1, deploy_config.OBSERVATION_DIM],
        )
        self.assertEqual(
            [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim],
            [1, deploy_config.ACTION_DIM],
        )

        metadata = {entry.key: entry.value for entry in model.metadata_props}
        self.assertEqual(metadata["run_path"], run_path)
        self.assertEqual(
            set(metadata),
            set(deploy_config.onnx_metadata(run_path)),
        )

        module = build_export_module(self.checkpoint)
        probe = np.random.default_rng(0).normal(size=(1, deploy_config.OBSERVATION_DIM)).astype(np.float32)
        with torch.no_grad():
            expected = module(torch.from_numpy(probe)).numpy()
        actual = np.asarray(ReferenceEvaluator(model).run(None, {"obs": probe})[0])
        np.testing.assert_allclose(actual, expected, atol=deploy_config.NORMALIZER_EPSILON * 1.0e-3)

    def test_missing_normalizer_is_rejected(self):
        checkpoint = self._checkpoint()
        del checkpoint["obs_norm_state_dict"]
        path = self.root / "missing_normalizer.pt"
        self._write_checkpoint(checkpoint, path)

        with self.assertRaisesRegex(ValueError, "no obs_norm_state_dict"):
            build_export_module(path)

    def test_negative_normalizer_std_is_rejected(self):
        checkpoint = self._checkpoint()
        checkpoint["obs_norm_state_dict"]["_std"][0, 0] = -1.0
        path = self.root / "negative_std.pt"
        self._write_checkpoint(checkpoint, path)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_export_module(path)

    def test_wrong_observation_width_is_rejected(self):
        checkpoint = self._checkpoint()
        state = checkpoint["model_state_dict"]
        state["actor.0.weight"] = state["actor.0.weight"][:, :-1]
        path = self.root / "wrong_width.pt"
        self._write_checkpoint(checkpoint, path)

        with self.assertRaisesRegex(ValueError, "only supports the 141-wide"):
            build_export_module(path)

    def test_nonstandard_actor_topology_is_rejected(self):
        checkpoint = self._checkpoint()
        state = checkpoint["model_state_dict"]
        state["actor.1.weight"] = state.pop("actor.2.weight")
        state["actor.1.bias"] = state.pop("actor.2.bias")
        path = self.root / "nonstandard.pt"
        self._write_checkpoint(checkpoint, path)

        with self.assertRaisesRegex(ValueError, "standard RSL-RL MLP topology"):
            build_export_module(path)

    def test_nonfinite_actor_is_rejected(self):
        checkpoint = self._checkpoint()
        checkpoint["model_state_dict"]["actor.0.weight"][0, 0] = torch.nan
        path = self.root / "nan.pt"
        self._write_checkpoint(checkpoint, path)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_export_module(path)

    def test_export_refuses_to_overwrite_checkpoint(self):
        before = self.checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError, "differ from the checkpoint"):
            export_policy(self.checkpoint, self.checkpoint)
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_failed_export_preserves_existing_output(self):
        self.output.write_bytes(b"existing")
        checkpoint = self._checkpoint()
        del checkpoint["obs_norm_state_dict"]
        bad = self.root / "bad.pt"
        self._write_checkpoint(checkpoint, bad)

        with self.assertRaisesRegex(ValueError, "no obs_norm_state_dict"):
            export_policy(bad, self.output)
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_standalone_cli_runs_outside_repository(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = ""
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL_DIR / "export_motion_policy.py"),
                "--checkpoint",
                str(self.checkpoint),
                "--output",
                str(self.output),
                "--run-path",
                "standalone/test",
            ],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.output.is_file())
        metadata = {entry.key: entry.value for entry in onnx.load(self.output).metadata_props}
        self.assertEqual(metadata["run_path"], "standalone/test")


if __name__ == "__main__":
    unittest.main()
