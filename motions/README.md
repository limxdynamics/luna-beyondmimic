# Bundled motion data

[English](README.md) | [中文](README.zh-CN.md)

## Source and status

The motion source represented by the `.npy` file in this directory was **collected internally by
LimX Dynamics**. The file in this repository is a processed kinematic reference trajectory
retargeted onto Luna; it is not the raw capture recording. The project owner confirms that all
represented data has been authorized for use.

## License

The bundled motion data carries no terms of its own. It is LimX Dynamics' own material and is
released under [Apache-2.0](../LICENSES/Apache-2.0.txt), the same license as the rest of LimX
Dynamics' contributions to this repository. Commercial use is permitted and needs no separate
authorization; the attribution requirement is the one Apache-2.0 already imposes through
[`NOTICE`](../NOTICE).

## Processing and schema

The source motion was retargeted onto the 27-DoF Luna input representation. This is the format
[`scripts/prepare_motion.py`](../scripts/prepare_motion.py) consumes: it solves the achilles
linkage, converts to the 51-DoF closed-loop training layout, holds the final pose and cross-checks
the result against forward kinematics, all into a temporary `.npz` that is never committed.

The archive is a pickled NumPy dictionary sampled at 50 Hz containing:

- `fps`
- `dof_names`, `dof_pos_vel` — `(frames, 27, 2)`, position and velocity per joint
- `body_names`, `body_states` — `(frames, 28, 13)`, position, quaternion, linear and angular
  velocity per body

The arrays contain robot joint/body kinematics. No images, audio, names, account identifiers, or
capture-site metadata are stored as named fields in this archive. This technical observation does
not replace the outstanding privacy/biometric review of the motion itself.

## Artifact inventory

| File | Frames | DoF | Duration | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `HU_L04_01_WAKAWAKA_an03.npy` | 1145 | 27 | 22.90 s | `bd9cc651c8daaacd43c0851232d7622723f0d8d74d27b09f86be501a341961e0` |

### Geometry provenance

This file was retargeted against a neighbouring revision of the Luna description, not against the
[`luna-description`](https://github.com/limx-luna/luna-description) commit this repository pins.
The two agree on every joint name and on the joint limits, but their link lengths differ by up to
2.5 mm — the largest single difference is at the elbow, and the deviation accumulates to a few
millimetres along the arm and leg chains.

The practical consequence is that
[`scripts/verify_motion_npz.py`](../scripts/verify_motion_npz.py) defaults to a 5 mm per-body
position tolerance rather than 1 mm. Use `--pos_tol 1e-3` for a motion you retargeted against the
pinned description.

## Recommended traceability fields

The following details are not included in the public repository. Retaining them in the project's
internal provenance records is recommended:

- identity and date of each raw capture session;
- collection method and participant/performer category;
- evidence reference and scope for the confirmed use authorization;
- privacy/biometric assessment;
- ownership of any choreography or source performance represented by the motion;
- mapping from each processed artifact above to its raw capture and processing record.

The project owner's current statement establishes source, authorization for publication and use,
and the Apache-2.0 terms.

## Review status

On 2026-09-01, the project owner confirmed that the data review is complete and found no remaining
publication blocker. The confirmation covers:

- LimX Dynamics' internal collection and authority to publish the processed trajectories;
- authorization for the represented data and performances;
- the privacy/biometric and choreography/source-performance boundary;
- publication under the conditions in this notice.

The traceability fields above remain recommended internal records, not prerequisites left open by
this repository notice.
