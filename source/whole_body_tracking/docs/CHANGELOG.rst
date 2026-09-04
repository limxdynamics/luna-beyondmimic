Changelog
---------

This is the Isaac Lab extension changelog and is kept deliberately short. The complete record for
the repository, including governance, licensing and tooling changes, is ``CHANGELOG.md`` at the
repository root.

0.1.0 (unreleased)
~~~~~~~~~~~~~~~~~

    10|Added
^^^^^

* Luna (HU_L04) closed-loop articulation support, registered as the single task
  ``Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0``.
* Deployable 141-dimensional policy observation and 27-dimensional motor action contract with
  projected gravity.
* Retargeted motion preparation: closed-loop linkage solve, 50 Hz conversion, held final pose,
  and forward-kinematics verification.
* CPU-only policy ONNX, DanceLibrary reference, and deployment parameter export.
    20|* Motion data cards, third-party notices, hardware disclosure and community governance.
* CPU-only GitHub CI and a complete unit-test suite.

Changed
^^^^^^^

* Training, replay, and conversion now use local motion files only; the WandB integration and
  upload scripts were removed.
* The robot description is referenced as the ``robot_descriptions/luna-description`` submodule
  rather than vendored into this package.
    30|* The package is explicitly source-only and supports editable installation rather than wheel,
  sdist, or PyPI distribution.

Upstream
^^^^^^^^

* Based on ``HybridRobotics/whole_body_tracking`` commit
  ``cd65172032893724b445448818c34165846d847d``. See ``THIRD_PARTY_NOTICES.md`` for attribution and
  license boundaries.
