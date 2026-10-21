# Changelog

This file records notable changes to this repository. It does not restate the upstream history of
[HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking), which
this project forked at `cd65172`; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the
fork record.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/). Until a `1.0.0` tag exists, the public interface —
task ID, ONNX metadata layout, reference TXT columns and the exported `param.yaml` — may change
between releases.

`source/whole_body_tracking/docs/CHANGELOG.rst` is the Isaac Lab extension's own changelog and
covers only that package. This file is the complete record.

## [Unreleased]

Initial public release of the LimX Dynamics fork, carrying version `0.1.0` in
`source/whole_body_tracking/config/extension.toml`. No release tag has been cut yet.

Relative to the upstream baseline:

### Added

- README banner rendered from the bundled dance motion, plus Highlights, Pipeline and
  Documentation sections at the top of `README.md` and `README.zh-CN.md`.
- Closed-loop simulation of the ankle and waist achilles linkages instead of a direct-drive
  approximation. The articulation carries 51 DoF while the action space stays 27 wide, six of
  which are motor angles rather than the joint they are named after.
- Local-only motion preparation: `scripts/prepare_motion.py` solves the linkage, converts to the
  training layout, holds the final pose and cross-checks the result against forward kinematics.
  `scripts/verify_motion_npz.py`, `scripts/linkage_quality.py` and `scripts/replay_npz.py` cover
  verification and diagnostics.
- Offline ONNX export via `tools/offline_onnx_exporter/`, producing `policy.onnx`,
  `reference.txt`, a complete `LimxDanceSDKHumanoidController` `param.yaml` and
  `linkage_report.json` from one trusted retargeted `.npy`.
- Deployment cross-checks `scripts/check_deploy_onnx.py` and
  `scripts/verify_parallel_motor_convention.py`, both requiring caller-supplied hardware
  configuration rather than a copy bundled here.
- Direct checkpoint paths through `--checkpoint`; bare filenames and regexes keep the legacy
  `--load_run` lookup.
- One reference motion, `motions/HU_L04_01_WAKAWAKA_an03.npy` (1145 frames, 50 Hz).
- Governance and disclosure records: `NOTICE`, `THIRD_PARTY_NOTICES.md`, `SECURITY.md`,
  `SAFETY.md`, `HARDWARE_DISCLOSURE.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, bilingual
  READMEs and Quick Starts, and a CPU-only GitHub Actions workflow.

### Changed

- Retargeted from HU_L03 to HU_L04. The single supported task is
  `Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0`.
- The robot description is referenced as a submodule at `robot_descriptions/luna-description`
  rather than vendored, so a public clone does not depend on this repository's LFS endpoint.
- Every hardware number in `deploy_config.py` is derived from that pinned description — gains from
  the MJCF armature table, joint travel from the URDF, achilles-motor travel from the MJCF — and
  `tests/test_deploy_config_tracks_description.py` recomputes all of them to fail on drift.
- Motion registry, checkpoint lookup and artifact handling are local; the upstream Weights &
  Biases integration and CSV ingest paths were removed.

### Removed

- Robot and task variants that are not supported here (G1, SMPL, humanoid) and the upstream upload
  paths.

### Fixed

- 将 `head_pitch_link` 加入追踪 body，修复头部两个关节不跟随参考动作的问题；critic 观测由 327 维变为 336 维，此前训练的 checkpoint 无法再加载。

### Security

- The retargeted `.npy` path uses NumPy pickle loading before schema validation, which is an
  accepted residual risk limited to trusted files. See
  [`SECURITY.md`](SECURITY.md#accepted-trusted-input-risk).

### Notes

- No trained policy is distributed. Every deployable artifact is one the user produces.
- Validated on Ubuntu 22.04.5, Python 3.10.15, PyTorch 2.5.1+cu118, Isaac Sim 4.5.0.0 and
  Isaac Lab 2.1.0. Other combinations are untested.
