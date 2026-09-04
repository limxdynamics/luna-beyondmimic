# Robot safety boundary

## Scope

This repository provides motion-tracking training, replay, and offline export for one deployment
configuration of Luna (HU_L04):

`Tracking-Flat-HU-L04-Parallel-Deploy-Gravity-v0`

It does not provide a complete site safety system, emergency-stop implementation, fall-protection
system, workspace-clearance procedure, or operator certification program. It also ships no trained
policy: every artifact you deploy is one you produced yourself.

## Approved disclosure and default behavior

On 2026-09-01, the project owner confirmed that the existing safety disclosure is sufficient and
approved normal operation as the default:

- `tools/offline_onnx_exporter/param.yaml` intentionally ships `dry_run: false`, so a policy
  exported with it drives the joints as soon as the controller starts;
- a mandatory dry-run stage is not part of the supported workflow;
- the hardware parameters and interfaces listed in
  [`HARDWARE_DISCLOSURE.md`](HARDWARE_DISCLOSURE.md) may be published.

The third item follows that record rather than restating it. Its scope was re-approved on
2026-09-04, when `param.yaml` grew from a path-only template into the complete controller
configuration the exporter now copies.

That template also carries `replay_reference: false`. Setting either flag is a bring-up aid, not a
safety mechanism: `dry_run` suppresses joint commands entirely, and `replay_reference` drives the
joints straight from the reference, which will not keep the robot upright.

This decision records the intended public behavior. It is not a certification that every robot,
site, controller build, or user-created policy is safe.

## Responsibility boundary

Anyone deploying on real hardware is responsible for:

- verifying the exact robot, controller, firmware, model, reference, and parameter versions;
- providing site-specific emergency handling, fall protection, workspace clearance, and trained
  supervision;
- validating communication-loss, stale-state, invalid-output, transition, and stop behavior;
- confirming torque, speed, travel, support, and environmental limits for that deployment;
- stopping operation when actual behavior differs from the reviewed configuration.

The repository's checks establish narrower facts:

- unit tests check software contracts;
- simulation checks model behavior in the tested simulator configuration;
- `check_deploy_onnx.py` checks ONNX signature, metadata, and joint permutation;
- `verify_parallel_motor_convention.py` checks motor sign and zero conventions against a caller-
  supplied independent linkage configuration.

None of those results replaces a real-hardware risk assessment.

Report a vulnerability or unsafe supported behavior privately using [`SECURITY.md`](SECURITY.md).
