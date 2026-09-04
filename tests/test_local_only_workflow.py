import pathlib
import re
import unittest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOVED_INTEGRATION = "wand" + "b"


class LocalOnlyWorkflowTest(unittest.TestCase):
    def test_runtime_has_no_cloud_tracker_integration(self):
        runtime_roots = (REPOSITORY_ROOT / "scripts", REPOSITORY_ROOT / "source")
        offenders = []
        for root in runtime_roots:
            for path in root.rglob("*.py"):
                if REMOVED_INTEGRATION in path.read_text(encoding="utf-8").lower():
                    offenders.append(path.relative_to(REPOSITORY_ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_public_docs_do_not_advertise_removed_workflow(self):
        documents = [
            REPOSITORY_ROOT / "README.md",
            REPOSITORY_ROOT / "README.zh-CN.md",
            REPOSITORY_ROOT / "QuickStart.md",
            REPOSITORY_ROOT / "Install.md",
            REPOSITORY_ROOT / "QuickStart.zh-CN.md",
        ]
        offenders = [
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in documents
            if REMOVED_INTEGRATION in path.read_text(encoding="utf-8").lower()
        ]
        self.assertEqual(offenders, [])

    def test_training_and_playback_share_the_npy_preparation_path(self):
        for entry_point in ("train.py", "play.py"):
            with self.subTest(entry_point=entry_point):
                source = (REPOSITORY_ROOT / "scripts/rsl_rl" / entry_point).read_text(encoding="utf-8")
                self.assertIn("add_motion_preparation_args(parser)", source)
                self.assertIn("prepare_motion_file(args_cli, parser)", source)
                self.assertNotIn("--registry_name", source)

        helper = (REPOSITORY_ROOT / "scripts/rsl_rl/motion_preparation.py").read_text(encoding="utf-8")
        self.assertIn('"--motion_file"', helper)
        self.assertIn('parser.error("--motion_file is required")', helper)
        self.assertIn(".npz is an internal intermediate", helper)
        self.assertIn("prepare_motion.py", helper)
        self.assertIn('"hu_l04_parallel"', helper)

    def test_combined_export_prepares_the_same_npy_input(self):
        exporter = (REPOSITORY_ROOT / "tools/offline_onnx_exporter/export_policy_and_reference.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--motion"', exporter)
        self.assertIn("prepare_motion.py", exporter)
        self.assertIn('"hu_l04_parallel"', exporter)
        self.assertIn('Path(scratch) / "motion.npz"', exporter)
        self.assertIn("convert_npz_to_txt(intermediate", exporter)

    def test_setup_does_not_install_cloud_tracker(self):
        setup_source = (REPOSITORY_ROOT / "source/whole_body_tracking/setup.py").read_text(encoding="utf-8")
        self.assertNotIn(REMOVED_INTEGRATION, setup_source.lower())

    def test_editable_install_does_not_upgrade_isaaclab_onnx_stack(self):
        setup_source = (REPOSITORY_ROOT / "source/whole_body_tracking/setup.py").read_text(encoding="utf-8")
        self.assertNotIn("onnxscript", setup_source)
        requirements = (REPOSITORY_ROOT / "tools/offline_onnx_exporter/requirements.txt").read_text(encoding="utf-8")
        self.assertIn("numpy==1.26.4", requirements)
        self.assertIn("onnx==1.16.1", requirements)

    def test_source_only_distribution_is_explicit(self):
        setup_source = (REPOSITORY_ROOT / "source/whole_body_tracking/setup.py").read_text(encoding="utf-8")
        self.assertIn('"Private :: Do Not Upload"', setup_source)
        for relative_path in (
            "README.md",
            "README.zh-CN.md",
            "Install.md",
            "QuickStart.md",
            "QuickStart.zh-CN.md",
        ):
            with self.subTest(path=relative_path):
                text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8").lower()
                self.assertIn("install -e", text)
                self.assertIn("wheel", text)

    def test_public_package_metadata_is_not_the_extension_template(self):
        extension = (REPOSITORY_ROOT / "source/whole_body_tracking/config/extension.toml").read_text(encoding="utf-8")
        self.assertIn('title = "LimxMimic — Luna Motion Tracking"', extension)
        self.assertIn('author = "LimX Dynamics"', extension)
        self.assertIn('maintainer = "LimX Dynamics"', extension)
        self.assertIn('repository = "https://github.com/limx-luna/luna-beyondmimic"', extension)
        self.assertNotIn("Extension Template", extension)
        self.assertNotIn("IsaacLabExtensionTemplate", extension)

    def test_deployment_checks_require_explicit_portable_config_paths(self):
        checker = (REPOSITORY_ROOT / "scripts/check_deploy_onnx.py").read_text(encoding="utf-8")
        convention = (REPOSITORY_ROOT / "scripts/verify_parallel_motor_convention.py").read_text(encoding="utf-8")
        # These scripts cross-check an export against configuration the caller supplies, so a
        # baked-in machine path would both break for everyone else and leak whoever wrote it.
        for home_root in ("/home/", "/Users/", "/root/"):
            self.assertNotIn(home_root, checker + convention)
        self.assertIn('"--hardware_config"', checker)
        self.assertIn("required=True", checker)
        self.assertIn('"--config_dir"', convention)
        self.assertIn('parser.error("--config_dir is required")', convention)

    def test_public_scope_contains_one_luna_task_and_closed_loop_assets(self):
        unsupported = (
            "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1",
            "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/humanoid",
            "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/hu_l04/flat_env_cfg.py",
            "source/whole_body_tracking/whole_body_tracking/robots/g1.py",
            "source/whole_body_tracking/whole_body_tracking/robots/smpl.py",
            "source/whole_body_tracking/whole_body_tracking/robots/hu_l04.py",
            "scripts/csv_to_npz.py",
        )
        for relative_path in unsupported:
            with self.subTest(path=relative_path):
                path = REPOSITORY_ROOT / relative_path
                if path.is_dir():
                    self.assertFalse(any(path.rglob("*.py")))
                else:
                    self.assertFalse(path.exists())

        task_source = (
            REPOSITORY_ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/hu_l04/__init__.py"
        ).read_text(encoding="utf-8")
        registered = re.findall(r'id="([^"]+)"', task_source)
        self.assertEqual(registered, ["Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0"])

        motion_files = sorted(path.name for path in (REPOSITORY_ROOT / "motions").glob("*.npy"))
        self.assertEqual(len(motion_files), 1)
        self.assertFalse(any((REPOSITORY_ROOT / "motions").glob("*.npz")))

    def test_robot_description_is_a_submodule_rather_than_vendored(self):
        modules = (REPOSITORY_ROOT / ".gitmodules").read_text(encoding="utf-8")
        self.assertIn("path = robot_descriptions/luna-description", modules)
        self.assertIn("url = https://github.com/limx-luna/luna-description", modules)

        # An uninitialised submodule is an empty directory, so the description may legitimately be
        # absent here. What must hold either way is that nothing is vendored under the package.
        asset_root = REPOSITORY_ROOT / "source/whole_body_tracking/whole_body_tracking/assets"
        vendored = sorted(path.name for path in asset_root.iterdir() if path.name != "__pycache__")
        self.assertEqual(vendored, ["__init__.py"])

        robot_source = (
            REPOSITORY_ROOT / "source/whole_body_tracking/whole_body_tracking/robots/hu_l04_parallel.py"
        ).read_text(encoding="utf-8")
        self.assertIn("luna_description_path", robot_source)
        self.assertIn('"HU_L04_description", "usd_parallel", "HU_L04_01.usd"', robot_source)


if __name__ == "__main__":
    unittest.main()
