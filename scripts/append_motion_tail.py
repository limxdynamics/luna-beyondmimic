"""Hold a motion's final pose for a while by appending a static tail to its motion.npz.

The tracking task never lets a policy stand still at the end of a reference. When the clip runs
out, ``MotionCommand._update_command`` teleports the environment to a fresh start time, so the
final pose exists for a frame or two and is never held. Deployment does the opposite: it repeats
the last frame until an operator switches controller, which is a state the policy was never
trained for and often cannot hold.

Appending the tail here rather than at load time keeps training and deployment on the same data.
The reference TXT that ``tools/offline_onnx_exporter`` writes is built from this same npz, so a
tailed motion makes the deployed "repeat the last frame" behaviour identical to what was trained.

The tail freezes the final pose and sets every velocity to zero, which is what standing still
means. A motion still moving at its last frame therefore gets an abrupt stop, so this warns when
the final velocities are too large for that to be a fair request.

.. code-block:: bash

    python scripts/append_motion_tail.py --input motions/x.npz --output motions/x_tail.npz \
        --tail_seconds 10
"""

import argparse
import numpy as np
import os

# Written back unchanged; the loader indexes them by frame, so every one has to grow together.
POSE_KEYS = ("joint_pos", "body_pos_w", "body_quat_w")
VELOCITY_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")

# Above this the final frame is moving enough that freezing it is a step change rather than a
# continuation. Angular velocity is allowed a little more room because a visually settled
# wrist/head at 30 Hz can retain a small final orientation delta while the base is already still.
STATIC_JOINT_VEL = 0.1  # rad/s
STATIC_BODY_LIN_VEL = 0.1  # m/s
STATIC_BODY_ANG_VEL = 0.2  # rad/s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a static tail to a motion.npz.")
    parser.add_argument("--input", type=str, required=True, help="Source motion.npz.")
    parser.add_argument("--output", type=str, required=True, help="Where to write the tailed motion.")
    parser.add_argument(
        "--tail_seconds",
        type=float,
        required=True,
        help=(
            "How long to hold the final pose. An episode is 10 s, so that much lets a policy spend"
            " a whole episode holding and is the point past which more adds little."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Append even if the motion is still moving at its last frame."
    )
    return parser.parse_args()


def check_final_frame_is_static(motion: dict, force: bool) -> None:
    joint_vel = np.abs(motion["joint_vel"][-1]).max()
    lin_vel = np.abs(motion["body_lin_vel_w"][-1]).max()
    ang_vel = np.abs(motion["body_ang_vel_w"][-1]).max()
    moving = joint_vel > STATIC_JOINT_VEL or lin_vel > STATIC_BODY_LIN_VEL or ang_vel > STATIC_BODY_ANG_VEL
    print(
        f"[INFO] Final frame speeds: joint {joint_vel:.4f} rad/s, body linear {lin_vel:.4f} m/s, "
        f"body angular {ang_vel:.4f} rad/s."
    )
    if not moving:
        return
    message = (
        "The motion is still moving at its last frame, so a frozen tail asks the robot to stop"
        " dead. Trim the motion to a resting pose, or pass --force to accept the discontinuity."
    )
    if not force:
        raise SystemExit(f"[ERROR] {message}")
    print(f"[WARN] {message}")


def append_tail(motion: dict, tail_frames: int) -> dict:
    out = {"fps": motion["fps"]}
    for key in POSE_KEYS:
        held = np.repeat(motion[key][-1:], tail_frames, axis=0)
        out[key] = np.concatenate([motion[key], held], axis=0)
    for key in VELOCITY_KEYS:
        still = np.zeros((tail_frames, *motion[key].shape[1:]), dtype=motion[key].dtype)
        out[key] = np.concatenate([motion[key], still], axis=0)
    return out


def main() -> None:
    args = parse_args()
    if args.tail_seconds <= 0:
        raise SystemExit("--tail_seconds must be positive.")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise SystemExit("--output must differ from --input.")

    with np.load(args.input) as data:
        missing = sorted({*POSE_KEYS, *VELOCITY_KEYS, "fps"} - set(data.files))
        if missing:
            raise SystemExit(f"[ERROR] {args.input} is missing {missing}.")
        motion = {key: data[key] for key in data.files}

    fps = int(np.asarray(motion["fps"]).reshape(-1)[0])
    num_frames = motion["joint_pos"].shape[0]
    for key in (*POSE_KEYS, *VELOCITY_KEYS):
        if motion[key].shape[0] != num_frames:
            raise SystemExit(f"[ERROR] {key} has {motion[key].shape[0]} frames, expected {num_frames}.")

    check_final_frame_is_static(motion, args.force)

    tail_frames = int(round(args.tail_seconds * fps))
    if tail_frames < 1:
        raise SystemExit(f"--tail_seconds {args.tail_seconds} rounds to zero frames at {fps} fps.")

    out = append_tail(motion, tail_frames)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output, **out)

    total = num_frames + tail_frames
    print(
        f"[INFO] {num_frames} frames ({num_frames / fps:.2f} s) + {tail_frames} held "
        f"({tail_frames / fps:.2f} s) = {total} frames ({total / fps:.2f} s)."
    )
    print(f"[INFO] Wrote {args.output}")


if __name__ == "__main__":
    main()
