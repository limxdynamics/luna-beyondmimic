import importlib
import numpy as np
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

TOOL_DIR = Path(__file__).parents[1] / "tools" / "offline_onnx_exporter"
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import deploy_config  # noqa: E402
from reference_converter import convert_npz_to_txt  # noqa: E402

# Imported rather than copied: these must track the articulation, and a stale duplicate here
# would let a mismatch pass the tests and only surface at training time.
_npy_to_npz = importlib.import_module("npy_to_npz")
HU_L04_PARALLEL_JOINT_ORDER = _npy_to_npz.HU_L04_PARALLEL_JOINT_ORDER
HU_L04_PARALLEL_BODY_ORDER = _npy_to_npz.HU_L04_PARALLEL_BODY_ORDER


class NpzMotionReferenceConversionTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.motion = self.root / "motion_parallel_tail10.npz"
        self.reference_output = self.root / "reference.txt"

        frames = 2
        joint_pos = np.zeros((frames, len(HU_L04_PARALLEL_JOINT_ORDER)), dtype=np.float32)
        joint_vel = np.zeros_like(joint_pos)
        joint_pos[0] = np.arange(len(HU_L04_PARALLEL_JOINT_ORDER), dtype=np.float32) / 1000.0
        joint_vel[0] = joint_pos[0] + 0.5

        body_pos = np.zeros((frames, len(HU_L04_PARALLEL_BODY_ORDER), 3), dtype=np.float32)
        body_quat = np.zeros((frames, len(HU_L04_PARALLEL_BODY_ORDER), 4), dtype=np.float32)
        body_lin_vel = np.zeros_like(body_pos)
        body_ang_vel = np.zeros_like(body_pos)
        body_quat[:, :, 0] = 1.0

        base_index = HU_L04_PARALLEL_BODY_ORDER.index(deploy_config.BASE_BODY_NAME)
        base_pos = np.array([1.0, 2.0, 0.8], dtype=np.float32)
        body_pos[0, base_index] = base_pos
        body_lin_vel[0, base_index] = [1.0, 2.0, 3.0]
        body_ang_vel[0, base_index] = [4.0, 5.0, 6.0]

        for offset, name in enumerate(deploy_config.KEY_BODY_NAMES, start=1):
            body_index = HU_L04_PARALLEL_BODY_ORDER.index(name)
            body_pos[0, body_index] = base_pos + np.array([offset, 0.1 * offset, -0.2 * offset], dtype=np.float32)

        np.savez(
            self.motion,
            fps=np.array([deploy_config.REFERENCE_FPS]),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=body_lin_vel,
            body_ang_vel_w=body_ang_vel,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_reference_txt_is_built_from_training_npz_motion(self):
        convert_npz_to_txt(self.motion, self.reference_output)

        rows = np.loadtxt(self.reference_output, ndmin=2)
        self.assertEqual(rows.shape, (2, deploy_config.REFERENCE_COLUMN_COUNT))

        joint_index = {name: index for index, name in enumerate(HU_L04_PARALLEL_JOINT_ORDER)}
        expected_sources = deploy_config.REFERENCE_TRAINING_JOINT_NAMES
        expected_joint_pos = [joint_index[name] / 1000.0 for name in expected_sources]
        np.testing.assert_allclose(rows[0, 0:27], expected_joint_pos, atol=1.0e-6)
        np.testing.assert_allclose(rows[0, 27:54], np.asarray(expected_joint_pos) + 0.5, atol=1.0e-6)
        np.testing.assert_allclose(rows[0, 54:67], [1.0, 2.0, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(
            rows[0, 67:].reshape(len(deploy_config.KEY_BODY_NAMES), 7)[:, 3:],
            np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(deploy_config.KEY_BODY_NAMES), 1)),
            atol=1.0e-6,
        )

    def test_waist_reference_limits_use_training_motor_space(self):
        with np.load(self.motion) as source:
            arrays = {key: source[key] for key in source.files}

        joint_index = {name: index for index, name in enumerate(HU_L04_PARALLEL_JOINT_ORDER)}
        arrays["joint_pos"][:, joint_index["waist_A_joint"]] = -0.70
        arrays["joint_pos"][:, joint_index["waist_B_joint"]] = -0.65
        np.savez(self.motion, **arrays)

        convert_npz_to_txt(self.motion, self.reference_output)

        rows = np.loadtxt(self.reference_output, ndmin=2)
        waist_roll_column = deploy_config.REFERENCE_JOINT_NAMES.index("waist_roll_joint")
        waist_pitch_column = deploy_config.REFERENCE_JOINT_NAMES.index("waist_pitch_joint")
        np.testing.assert_allclose(rows[:, waist_roll_column], -0.70, atol=1.0e-6)
        np.testing.assert_allclose(rows[:, waist_pitch_column], -0.65, atol=1.0e-6)


class CombinedExportOutputPathTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.checkpoint = self.root / "model_14999.pt"
        self.motion = self.root / "motion.npy"
        self.output_path = self.root / "exported"
        self.policy_output = self.output_path / "policy.onnx"
        self.reference_output = self.output_path / "reference.txt"
        self.checkpoint.write_text("checkpoint", encoding="utf-8")
        self.motion.write_text("motion", encoding="utf-8")

    def tearDown(self):
        sys.modules.pop("export_policy_and_reference", None)
        self.temp_dir.cleanup()

    def test_combined_cli_writes_policy_and_reference_under_output_path(self):
        calls = []
        module = self._load_combined_exporter_with_fakes(calls)

        def prepare_motion(
            source,
            intermediate,
            *,
            tail_seconds,
            max_unreachable_error,
            strict_motion_check,
            report_output,
        ):
            calls.append(
                (
                    "prepare",
                    Path(source),
                    tail_seconds,
                    max_unreachable_error,
                    strict_motion_check,
                    Path(report_output),
                )
            )
            Path(intermediate).write_text("intermediate", encoding="utf-8")
            Path(report_output).write_text("{}", encoding="utf-8")

        with (
            mock.patch.object(module, "_prepare_motion", side_effect=prepare_motion),
            mock.patch.object(
                sys,
                "argv",
                [
                    "export_policy_and_reference.py",
                    "--checkpoint",
                    str(self.checkpoint),
                    "--motion",
                    str(self.motion),
                    "--output_path",
                    str(self.output_path),
                ],
            ),
        ):
            result = module.main()

        self.assertEqual(result, 0)
        self.assertEqual(self.policy_output.read_text(encoding="utf-8"), "policy")
        self.assertEqual(self.reference_output.read_text(encoding="utf-8"), "reference")
        self.assertEqual(
            (self.output_path / "param.yaml").read_text(encoding="utf-8"),
            (TOOL_DIR / "param.yaml").read_text(encoding="utf-8"),
        )
        param = (self.output_path / "param.yaml").read_text(encoding="utf-8")
        self.assertIn('policy_dir: "/policy/test"', param)
        self.assertIn('ref_dir: "motion/test.txt"', param)
        self.assertIn("user_torque_limit:", param)
        self.assertIn('joint_space: "motor"', param)
        self.assertEqual(
            calls[0],
            ("policy", self.checkpoint, self.policy_output, deploy_config.DEFAULT_RUN_PATH),
        )
        self.assertEqual(
            calls[1],
            (
                "prepare",
                self.motion,
                10.0,
                None,
                False,
                self.output_path / "linkage_report.json",
            ),
        )
        self.assertEqual(calls[2][0], "reference")
        self.assertEqual(calls[2][1].name, "motion.npz")
        self.assertEqual(calls[2][2], self.reference_output)
        self.assertFalse((self.output_path / "motion.npz").exists())
        self.assertTrue((self.output_path / "linkage_report.json").is_file())

    def test_cli_without_motion_exports_only_policy(self):
        calls = []
        module = self._load_combined_exporter_with_fakes(calls)
        self.output_path.mkdir()
        for name in ("policy.onnx", "reference.txt", "param.yaml", "linkage_report.json"):
            (self.output_path / name).write_text("stale", encoding="utf-8")
        deprecated_policy = self.output_path / "policy" / "test" / "policy.onnx"
        deprecated_reference = self.output_path / "motion" / "test.txt"
        deprecated_policy.parent.mkdir(parents=True)
        deprecated_reference.parent.mkdir(parents=True)
        deprecated_policy.write_text("stale", encoding="utf-8")
        deprecated_reference.write_text("stale", encoding="utf-8")

        with mock.patch.object(
            sys,
            "argv",
            [
                "export_policy_and_reference.py",
                "--checkpoint",
                str(self.checkpoint),
                "--output_path",
                str(self.output_path),
            ],
        ):
            result = module.main()

        self.assertEqual(result, 0)
        self.assertEqual(
            calls,
            [("policy", self.checkpoint, self.policy_output, deploy_config.DEFAULT_RUN_PATH)],
        )
        self.assertTrue(self.policy_output.is_file())
        self.assertFalse(self.reference_output.exists())
        self.assertFalse((self.output_path / "param.yaml").exists())
        self.assertFalse((self.output_path / "linkage_report.json").exists())
        files = sorted(
            path.relative_to(self.output_path).as_posix() for path in self.output_path.rglob("*") if path.is_file()
        )
        self.assertEqual(files, ["policy.onnx"])

    def test_motion_failure_keeps_policy_and_removes_stale_bundle_outputs(self):
        calls = []
        module = self._load_combined_exporter_with_fakes(calls)
        self.output_path.mkdir()
        for name in ("policy.onnx", "reference.txt", "param.yaml", "linkage_report.json"):
            (self.output_path / name).write_text("stale", encoding="utf-8")
        deprecated_policy = self.output_path / "policy" / "test" / "policy.onnx"
        deprecated_reference = self.output_path / "motion" / "test.txt"
        deprecated_policy.parent.mkdir(parents=True)
        deprecated_reference.parent.mkdir(parents=True)
        deprecated_policy.write_text("stale", encoding="utf-8")
        deprecated_reference.write_text("stale", encoding="utf-8")

        def fail_preparation(*args, **kwargs):
            Path(kwargs["report_output"]).write_text('{"strict_check_passed": false}', encoding="utf-8")
            raise RuntimeError("linkage residual exceeds limit")

        with (
            mock.patch.object(module, "_prepare_motion", side_effect=fail_preparation),
            mock.patch.object(
                sys,
                "argv",
                [
                    "export_policy_and_reference.py",
                    "--checkpoint",
                    str(self.checkpoint),
                    "--motion",
                    str(self.motion),
                    "--output_path",
                    str(self.output_path),
                    "--strict-motion-check",
                ],
            ),
        ):
            result = module.main()

        self.assertEqual(result, 1)
        self.assertEqual(self.policy_output.read_text(encoding="utf-8"), "policy")
        self.assertFalse(self.reference_output.exists())
        self.assertFalse((self.output_path / "param.yaml").exists())
        self.assertTrue((self.output_path / "linkage_report.json").is_file())
        self.assertFalse(deprecated_policy.exists())
        self.assertFalse(deprecated_reference.exists())

    def test_explicit_residual_limit_enables_strict_motion_check(self):
        calls = []
        module = self._load_combined_exporter_with_fakes(calls)

        def prepare_motion(source, intermediate, **kwargs):
            calls.append(kwargs)
            Path(intermediate).write_text("intermediate", encoding="utf-8")
            Path(kwargs["report_output"]).write_text("{}", encoding="utf-8")

        with (
            mock.patch.object(module, "_prepare_motion", side_effect=prepare_motion),
            mock.patch.object(
                sys,
                "argv",
                [
                    "export_policy_and_reference.py",
                    "--checkpoint",
                    str(self.checkpoint),
                    "--motion",
                    str(self.motion),
                    "--output_path",
                    str(self.output_path),
                    "--motion-max-unreachable-error",
                    "0.19",
                ],
            ),
        ):
            result = module.main()

        self.assertEqual(result, 0)
        self.assertTrue(calls[1]["strict_motion_check"])
        self.assertEqual(calls[1]["max_unreachable_error"], 0.19)

    def test_prepare_rejects_an_npz_user_input(self):
        module = self._load_combined_exporter_with_fakes([])
        npz_motion = self.root / "motion.npz"
        npz_motion.write_bytes(b"not used")

        with self.assertRaisesRegex(ValueError, "trusted retargeted .npy"):
            module._prepare_motion(
                npz_motion,
                self.root / "intermediate.npz",
                tail_seconds=10.0,
                max_unreachable_error=None,
                strict_motion_check=False,
                report_output=self.root / "linkage_report.json",
            )

    @staticmethod
    def _load_combined_exporter_with_fakes(calls):
        policy_exporter = types.ModuleType("policy_exporter")
        reference_converter = types.ModuleType("reference_converter")

        def export_policy(checkpoint_path, output_path, *, run_path):
            calls.append(("policy", Path(checkpoint_path), Path(output_path), run_path))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text("policy", encoding="utf-8")
            return Path(output_path)

        def convert_npz_to_txt(motion_path, output_path):
            calls.append(("reference", Path(motion_path), Path(output_path)))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text("reference", encoding="utf-8")
            return Path(output_path)

        policy_exporter.export_policy = export_policy
        reference_converter.convert_npz_to_txt = convert_npz_to_txt
        sys.modules.pop("export_policy_and_reference", None)
        with mock.patch.dict(
            sys.modules,
            {
                "policy_exporter": policy_exporter,
                "reference_converter": reference_converter,
            },
        ):
            return importlib.import_module("export_policy_and_reference")


if __name__ == "__main__":
    unittest.main()
