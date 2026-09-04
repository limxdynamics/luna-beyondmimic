"""Convert an xsens-retargeted HU_L04 motion (.npy) into the motion.npz consumed by the tracking task.

The input already carries per-body poses and velocities, so no forward kinematics is needed and
this script never launches Isaac Sim.

The closed-loop variant needs 24 joint values the retargeter does not produce - the achilles
motors and the rod joints - so for ``--robot hu_l04_parallel`` they are read from the file
``scripts/solve_parallel_linkage.py`` writes.

.. code-block:: bash

    python scripts/npy_to_npz.py --input_file motion_npy/motion.npy \
        --robot hu_l04_parallel --linkage_file /tmp/linkage.npz \
        --output_file motions/motion_parallel.npz
"""

import argparse
import numpy as np
import os
import tempfile

# Isaac Sim orders articulation links breadth-first after merging fixed joints, which does not
# match the URDF declaration order used by the .npy files. MotionLoader slices the npz body axis
# with indices obtained from the live articulation and uses joint_pos/joint_vel without any
# reindexing, so both axes below must follow the articulation.
#
# Dumped with scripts/dump_articulation_order.py. The closed-loop description carries the 27
# retargeted joint names plus 6 achilles motors and the 18 DoF of 6 spherical rod joints, and 12
# more bodies for the cranks and rods. Four retargeted names - the ankle pitch/roll and waist
# roll/pitch joints - are passive here, moved by the linkage rather than by a drive.
HU_L04_PARALLEL_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_A_joint",
    "waist_B_joint",
    "left_A_achilles_joint",
    "left_B_achilles_joint",
    "left_ankle_pitch_joint",
    "right_A_achilles_joint",
    "right_B_achilles_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_A_rod_joint:0",
    "waist_A_rod_joint:1",
    "waist_A_rod_joint:2",
    "waist_B_rod_joint:0",
    "waist_B_rod_joint:1",
    "waist_B_rod_joint:2",
    "left_A_achilles_rod_joint:0",
    "left_A_achilles_rod_joint:1",
    "left_A_achilles_rod_joint:2",
    "left_B_achilles_rod_joint:0",
    "left_B_achilles_rod_joint:1",
    "left_B_achilles_rod_joint:2",
    "left_ankle_roll_joint",
    "right_A_achilles_rod_joint:0",
    "right_A_achilles_rod_joint:1",
    "right_A_achilles_rod_joint:2",
    "right_B_achilles_rod_joint:0",
    "right_B_achilles_rod_joint:1",
    "right_B_achilles_rod_joint:2",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

HU_L04_PARALLEL_BODY_ORDER = (
    "base_link",
    # Fixed-joint IMU mount. It carries no DoF and the retargeter does not produce it, so it is
    # zero-filled like the other helper links - but it still occupies a slot in the articulation.
    "base_imu",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "waist_pitch_link",
    "left_knee_link",
    "right_knee_link",
    "head_yaw_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "waist_A_link",
    "waist_B_link",
    "left_A_achilles_link",
    "left_B_achilles_link",
    "left_ankle_pitch_link",
    "right_A_achilles_link",
    "right_B_achilles_link",
    "right_ankle_pitch_link",
    "head_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "waist_A_rod_link",
    "waist_B_rod_link",
    "left_A_achilles_rod_link",
    "left_B_achilles_rod_link",
    "left_ankle_roll_link",
    "right_A_achilles_rod_link",
    "right_B_achilles_rod_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "contact_foot_center_L",
    "contact_foot_heel_L",
    "contact_foot_tip_L",
    "contact_foot_center_R",
    "contact_foot_heel_R",
    "contact_foot_tip_R",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_hand_contact",
    "left_hand_manip",
    "right_hand_contact",
    "right_hand_manip",
)

ROBOT_ORDERS = {
    "hu_l04_parallel": (HU_L04_PARALLEL_JOINT_ORDER, HU_L04_PARALLEL_BODY_ORDER),
}

# 1 / (decimation * sim.dt) of TrackingEnvCfg. MotionLoader advances one row per environment
# step and ignores the stored fps, so the output rate has to match the control rate.
ENV_CONTROL_FPS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a retargeted .npy motion into motion.npz.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .npy motion.")
    parser.add_argument("--output_fps", type=int, default=ENV_CONTROL_FPS, help="Target sampling rate.")
    parser.add_argument(
        "--robot",
        type=str,
        default="hu_l04_parallel",
        choices=sorted(ROBOT_ORDERS),
        help="Supported articulation (fixed to the Luna closed-loop model).",
    )
    parser.add_argument(
        "--linkage_file",
        type=str,
        default=None,
        help="Output of solve_parallel_linkage.py. Required.",
    )
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Source frame range: START END (both inclusive, 1-based). Defaults to all frames.",
    )
    parser.add_argument("--output_file", type=str, default=None, help="Where to write the npz.")
    return parser.parse_args()


def describe_mismatch(label: str, actual: set[str], expected: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return f"{label} mismatch.\n  missing from motion: {missing}\n  unexpected in motion: {extra}"


def load_linkage(path: str, num_frames: int, fps: int) -> tuple[list[str], np.ndarray]:
    """Read the solved achilles DoF and differentiate them.

    The solver works frame by frame on a static mechanism, so it has no velocity to report. A
    central difference over the source frames recovers one; the motion is linearly resampled
    afterwards anyway, so nothing more careful is warranted.
    """
    data = np.load(path)
    names = [str(n) for n in data["joint_names"]]
    pos = np.asarray(data["joint_pos"], dtype=np.float64)
    if pos.shape[0] != num_frames:
        raise ValueError(f"{path} has {pos.shape[0]} frames but the motion has {num_frames}.")
    vel = np.gradient(pos, axis=0) * fps
    return names, np.stack([pos, vel], axis=-1)


def load_motion(path: str, frame_range: tuple[int, int] | None, robot: str, linkage_file: str | None) -> dict:
    joint_order, body_order = ROBOT_ORDERS[robot]
    data = np.load(path, allow_pickle=True).item()

    body_names = list(data["body_names"])
    dof_names = list(data["dof_names"])
    body_states = np.asarray(data["body_states"], dtype=np.float64)
    dof_pos_vel = np.asarray(data["dof_pos_vel"], dtype=np.float64)

    if body_states.shape[1] != len(body_names):
        raise ValueError(f"body_states has {body_states.shape[1]} bodies but body_names has {len(body_names)}.")
    if dof_pos_vel.shape[1] != len(dof_names):
        raise ValueError(f"dof_pos_vel has {dof_pos_vel.shape[1]} joints but dof_names has {len(dof_names)}.")

    if linkage_file is not None:
        linkage_names, linkage_pos_vel = load_linkage(linkage_file, dof_pos_vel.shape[0], int(data["fps"]))
        dof_names = dof_names + linkage_names
        dof_pos_vel = np.concatenate([dof_pos_vel, linkage_pos_vel], axis=1)
        print(f"[INFO] Merged {len(linkage_names)} solved linkage DoF from {linkage_file}.")

    # Every joint of the articulation must be accounted for: unlike the bodies below, the joint
    # vector is written back into the robot at every reset, so a zero-filled slot would be a
    # configuration the robot is physically told to adopt.
    if set(dof_names) != set(joint_order):
        raise ValueError(describe_mismatch("Joint name", set(dof_names), set(joint_order)))
    # The articulation carries fixed-joint helper links that the retargeter does not produce.
    # They are never read - MotionLoader only slices the bodies named in the env cfg - so they
    # are zero-filled below. Bodies present in the motion but absent from the articulation are
    # still an error, since that means the two describe different robots.
    unexpected = set(body_names) - set(body_order)
    if unexpected:
        raise ValueError(describe_mismatch("Body name", set(body_names), set(body_order)))

    if frame_range is not None:
        start, end = frame_range
        body_states = body_states[start - 1 : end]
        dof_pos_vel = dof_pos_vel[start - 1 : end]

    quat_norm = np.linalg.norm(body_states[:, :, 3:7], axis=-1)
    if np.abs(quat_norm - 1.0).max() > 1e-3:
        raise ValueError(f"Quaternions are not unit length (max deviation {np.abs(quat_norm - 1.0).max():.4g}).")

    dof_perm = [dof_names.index(n) for n in joint_order]
    dof_pos_vel = dof_pos_vel[:, dof_perm]

    reordered = np.zeros((body_states.shape[0], len(body_order), body_states.shape[2]))
    reordered[:, :, 6] = 1.0  # keep the unfilled slots' quaternions unit-length (xyzw)
    for slot, name in enumerate(body_order):
        if name in body_names:
            reordered[:, slot] = body_states[:, body_names.index(name)]
    missing = [n for n in body_order if n not in body_names]
    if missing:
        print(f"[INFO] Zero-filled {len(missing)} bodies absent from the motion: {missing}")
    body_states = reordered

    return {
        "fps": int(data["fps"]),
        "body_pos": body_states[:, :, 0:3],
        "body_quat_xyzw": body_states[:, :, 3:7],
        "body_lin_vel": body_states[:, :, 7:10],
        "body_ang_vel": body_states[:, :, 10:13],
        "joint_pos": dof_pos_vel[:, :, 0],
        "joint_vel": dof_pos_vel[:, :, 1],
    }


def body_z_axis_alignment(quat: np.ndarray, order: str) -> float:
    """Mean cosine between a body's own z axis and world +z, under the given component order.

    Only the x and y components enter the third diagonal term of the rotation matrix.
    """
    x_index, y_index = (0, 1) if order == "xyzw" else (1, 2)
    x, y = quat[:, x_index], quat[:, y_index]
    return float((1.0 - 2.0 * (x * x + y * y)).mean())


def check_quaternion_convention(root_quat_xyzw: np.ndarray) -> None:
    as_xyzw = body_z_axis_alignment(root_quat_xyzw, "xyzw")
    as_wxyz = body_z_axis_alignment(root_quat_xyzw, "wxyz")
    if as_wxyz > as_xyzw:
        print(
            "[WARN] The root orientation looks more upright when read as wxyz "
            f"(wxyz {as_wxyz:.4f} vs xyzw {as_xyzw:.4f}). The producer may have changed convention; "
            "verify before training."
        )
    else:
        print(f"[INFO] Quaternion convention check passed (xyzw {as_xyzw:.4f} vs wxyz {as_wxyz:.4f}).")


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def compute_frame_blend(num_frames: int, input_fps: int, output_fps: int):
    duration = (num_frames - 1) / input_fps
    times = np.arange(0.0, duration, 1.0 / output_fps)
    phase = times / duration
    index_0 = np.floor(phase * (num_frames - 1)).astype(np.int64)
    index_1 = np.minimum(index_0 + 1, num_frames - 1)
    blend = phase * (num_frames - 1) - index_0
    return index_0, index_1, blend


def lerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
    blend = blend.reshape(-1, *([1] * (a.ndim - 1)))
    return a * (1.0 - blend) + b * blend


def slerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Shortest-arc slerp over arrays of wxyz quaternions shaped (T, B, 4)."""
    blend = blend.reshape(-1, *([1] * (a.ndim - 1)))
    dot = (a * b).sum(axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot).clip(max=1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    # Fall back to lerp where the rotations are nearly identical and sin(theta) underflows.
    near = sin_theta < 1e-6
    w_a = np.where(near, 1.0 - blend, np.sin((1.0 - blend) * theta) / np.where(near, 1.0, sin_theta))
    w_b = np.where(near, blend, np.sin(blend * theta) / np.where(near, 1.0, sin_theta))

    out = w_a * a + w_b * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def resample(motion: dict, output_fps: int) -> dict:
    input_fps = motion["fps"]
    num_frames = motion["joint_pos"].shape[0]
    i0, i1, blend = compute_frame_blend(num_frames, input_fps, output_fps)

    out = {"fps": np.array([output_fps])}
    for key in ("body_pos", "body_lin_vel", "body_ang_vel", "joint_pos", "joint_vel"):
        out[key] = lerp(motion[key][i0], motion[key][i1], blend)
    quat_wxyz = quat_xyzw_to_wxyz(motion["body_quat_xyzw"])
    out["body_quat"] = slerp(quat_wxyz[i0], quat_wxyz[i1], blend)

    print(
        f"[INFO] Resampled {num_frames} frames @ {input_fps}fps -> "
        f"{out['joint_pos'].shape[0]} frames @ {output_fps}fps ({num_frames / input_fps:.2f}s)."
    )
    return out


def main() -> None:
    args = parse_args()

    if args.output_fps != ENV_CONTROL_FPS:
        print(
            f"[WARN] output_fps={args.output_fps} differs from the environment control rate "
            f"{ENV_CONTROL_FPS}Hz. The motion will play back at the wrong speed during training."
        )
    if args.linkage_file is None:
        raise SystemExit("--linkage_file is required.")

    joint_order, body_order = ROBOT_ORDERS[args.robot]
    motion = load_motion(
        args.input_file, tuple(args.frame_range) if args.frame_range else None, args.robot, args.linkage_file
    )
    print(f"[INFO] Loaded {args.input_file}: {motion['joint_pos'].shape[0]} frames @ {motion['fps']}fps.")
    print(f"[INFO] Target articulation {args.robot}: {len(joint_order)} joints, {len(body_order)} bodies.")
    check_quaternion_convention(motion["body_quat_xyzw"][:, body_order.index("base_link")])

    resampled = resample(motion, args.output_fps)

    output_file = args.output_file or os.path.join(tempfile.gettempdir(), "motion.npz")
    np.savez(
        output_file,
        fps=resampled["fps"],
        joint_pos=resampled["joint_pos"].astype(np.float32),
        joint_vel=resampled["joint_vel"].astype(np.float32),
        body_pos_w=resampled["body_pos"].astype(np.float32),
        body_quat_w=resampled["body_quat"].astype(np.float32),
        body_lin_vel_w=resampled["body_lin_vel"].astype(np.float32),
        body_ang_vel_w=resampled["body_ang_vel"].astype(np.float32),
    )
    print(f"[INFO] Wrote {output_file}")


if __name__ == "__main__":
    main()
