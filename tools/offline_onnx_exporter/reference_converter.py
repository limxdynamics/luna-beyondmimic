"""Convert a retargeted HU_L04 motion (.npy) into the DanceLibrary reference TXT.

The retargeter describes the serial robot: it gives ankle and waist angles, not the achilles
motors the closed-loop robot is actually driven by. Those six values are recovered here with the
deployment's own closed-form linkage model, so no simulator is involved and no intermediate file
has to be produced first.
"""

from __future__ import annotations

import numpy as np
import os
import stat
import tempfile
from pathlib import Path

try:
    from . import deploy_config
except ImportError:
    import deploy_config

# How far a stored quaternion may be from unit length. The retargeter writes float32, so exact
# unit length is not on offer.
QUATERNION_TOLERANCE = 1.0e-3

TEXT_DECIMALS = 6

# What the file may differ from the matrix by. Writing rounds, and a value sitting exactly halfway
# between two printable ones rounds by a full half digit, so the check is set one whole digit wide:
# it is there to catch a mangled or misaligned write, not to measure rounding.
TEXT_ROUNDING_TOLERANCE = 10.0**-TEXT_DECIMALS


def quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[..., 1:] *= -1.0
    return result


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply wxyz quaternions."""
    w1, x1, y1, z1 = (left[..., index] for index in range(4))
    w2, x2, y2, z2 = (right[..., index] for index in range(4))
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    zeros = np.zeros(vector.shape[:-1] + (1,), dtype=vector.dtype)
    vector_quaternion = np.concatenate([zeros, vector], axis=-1)
    return quat_mul(quat_mul(quat_conjugate(quaternion), vector_quaternion), quaternion)[..., 1:]


def yaw_from_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (quaternion[..., index] for index in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    half = 0.5 * yaw
    return np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=-1)


def _load_motion(path: Path) -> dict:
    """Read and check a retargeted motion.

    The file is a pickled dict, so only ever run this on motions produced by the retargeting
    pipeline: ``allow_pickle`` executes whatever the file says, and no amount of checking
    afterwards undoes that.
    """
    if path.suffix != ".npy":
        raise ValueError(f"Motion {path} must be a .npy file.")
    if not path.is_file():
        raise ValueError(f"No such motion: {path}")
    try:
        payload = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"Could not load motion {path}: {exc}") from exc
    if not isinstance(payload, np.ndarray) or payload.shape != ():
        raise ValueError(f"Motion {path} is not a retargeted motion dictionary.")
    motion = payload.item()
    if not isinstance(motion, dict):
        raise ValueError(f"Motion {path} is not a retargeted motion dictionary.")

    for key in ("body_states", "dof_pos_vel", "body_names", "dof_names", "fps"):
        if key not in motion:
            raise ValueError(f"Motion {path} has no '{key}' entry.")

    fps = motion["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float, np.integer, np.floating)):
        raise ValueError(f"Motion fps must be a number, got {type(fps).__name__}.")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Motion fps must be finite and positive, got {fps}.")

    body_names = _validated_names("body_names", motion["body_names"])
    joint_names = _validated_names("dof_names", motion["dof_names"])
    body_states = np.asarray(motion["body_states"], dtype=np.float64)
    joint_states = np.asarray(motion["dof_pos_vel"], dtype=np.float64)

    if body_states.ndim != 3 or body_states.shape[2] != 13:
        raise ValueError(f"Motion body_states must have shape (frames, bodies, 13), got {body_states.shape}.")
    if joint_states.ndim != 3 or joint_states.shape[2] != 2:
        raise ValueError(f"Motion dof_pos_vel must have shape (frames, joints, 2), got {joint_states.shape}.")
    if body_states.shape[0] != joint_states.shape[0]:
        raise ValueError(
            f"Motion body_states has {body_states.shape[0]} frames but dof_pos_vel has {joint_states.shape[0]}."
        )
    if body_states.shape[0] < 2:
        raise ValueError("Motion must contain at least two frames to define a velocity.")
    if body_states.shape[1] != len(body_names):
        raise ValueError(f"Motion has {body_states.shape[1]} bodies but {len(body_names)} body names.")
    if joint_states.shape[1] != len(joint_names):
        raise ValueError(f"Motion has {joint_states.shape[1]} joints but {len(joint_names)} joint names.")
    if not np.isfinite(body_states).all() or not np.isfinite(joint_states).all():
        raise ValueError("Motion contains values that are not finite.")

    missing_joints = [name for name in deploy_config.SOURCE_JOINT_NAMES if name not in joint_names]
    if missing_joints:
        raise ValueError(f"Motion is missing the joints {missing_joints}.")
    missing_bodies = [
        name for name in (deploy_config.BASE_BODY_NAME, *deploy_config.KEY_BODY_NAMES) if name not in body_names
    ]
    if missing_bodies:
        raise ValueError(f"Motion is missing the bodies {missing_bodies}.")

    quaternion_norm = np.linalg.norm(body_states[:, :, 3:7], axis=-1)
    if np.max(np.abs(quaternion_norm - 1.0)) > QUATERNION_TOLERANCE:
        raise ValueError("Motion body orientations are not unit quaternions.")

    return {
        "fps": fps,
        "body_index": {name: index for index, name in enumerate(body_names)},
        "joint_index": {name: index for index, name in enumerate(joint_names)},
        "body_states": body_states,
        "joint_states": joint_states,
    }


def _load_training_motion_npz(path: Path) -> dict:
    """Read and check the motion.npz consumed by the HU_L04 parallel training task."""
    if path.suffix != ".npz":
        raise ValueError(f"Training motion {path} must be a .npz file.")
    if not path.is_file():
        raise ValueError(f"No such training motion: {path}")

    required = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    try:
        with np.load(path) as data:
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"Training motion {path} is missing {missing}.")
            fps_value = np.asarray(data["fps"])
            arrays = {key: np.array(data[key], dtype=np.float64, copy=True) for key in required if key != "fps"}
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not load training motion {path}: {exc}") from exc

    if fps_value.size != 1:
        raise ValueError(f"Training motion fps must be a scalar, got shape {fps_value.shape}.")
    fps = float(fps_value.reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Training motion fps must be finite and positive, got {fps}.")
    if abs(fps - deploy_config.REFERENCE_FPS) > 1.0e-6:
        raise ValueError(f"Training motion must be {deploy_config.REFERENCE_FPS} Hz, got {fps:g} Hz.")

    joint_pos = arrays["joint_pos"]
    joint_vel = arrays["joint_vel"]
    expected_joint_count = len(deploy_config.TRAINING_JOINT_NAMES)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != expected_joint_count:
        raise ValueError(
            f"Training motion joint_pos must have shape (frames, {expected_joint_count}), got {joint_pos.shape}."
        )
    if joint_vel.shape != joint_pos.shape:
        raise ValueError(f"Training motion joint_vel has shape {joint_vel.shape}, expected {joint_pos.shape}.")
    frames = joint_pos.shape[0]
    if frames == 0:
        raise ValueError("Training motion has no frames.")

    expected_body_count = len(deploy_config.TRAINING_BODY_NAMES)
    body_shapes = {
        "body_pos_w": 3,
        "body_quat_w": 4,
        "body_lin_vel_w": 3,
        "body_ang_vel_w": 3,
    }
    for key, width in body_shapes.items():
        expected = (frames, expected_body_count, width)
        if arrays[key].shape != expected:
            raise ValueError(f"Training motion {key} has shape {arrays[key].shape}, expected {expected}.")

    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Training motion contains values that are not finite.")

    quaternion_norm = np.linalg.norm(arrays["body_quat_w"], axis=-1)
    if np.max(np.abs(quaternion_norm - 1.0)) > QUATERNION_TOLERANCE:
        raise ValueError("Training motion body orientations are not unit quaternions.")

    return {
        "fps": fps,
        "joint_index": {name: index for index, name in enumerate(deploy_config.TRAINING_JOINT_NAMES)},
        "body_index": {name: index for index, name in enumerate(deploy_config.TRAINING_BODY_NAMES)},
        **arrays,
    }


def _validated_names(label: str, names) -> list[str]:
    if not isinstance(names, (list, tuple, np.ndarray)):
        raise ValueError(f"Motion {label} must be a sequence of names.")
    values = [str(name) for name in names]
    if any(not name for name in values):
        raise ValueError(f"Motion {label} contains an empty name.")
    if len(set(values)) != len(values):
        raise ValueError(f"Motion {label} contains duplicates.")
    return values


def _solve_reference_joints(motion: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build the 27 reference joint positions and velocities in DanceLibrary order."""
    joint_index = motion["joint_index"]
    joint_states = motion["joint_states"]
    frames = joint_states.shape[0]

    positions: dict[str, np.ndarray] = {}
    velocities: dict[str, np.ndarray] = {}
    for name in deploy_config.DIRECT_JOINT_NAMES:
        column = joint_index[name]
        positions[name] = joint_states[:, column, 0]
        velocities[name] = joint_states[:, column, 1]

    for mechanism in deploy_config.MECHANISMS:
        roll = joint_states[:, joint_index[mechanism.roll_joint_name], 0]
        pitch = joint_states[:, joint_index[mechanism.pitch_joint_name], 0]
        motors = deploy_config.solve_motor_angles(mechanism, roll=roll, pitch=pitch)
        # The linkage model is positional, so the motor rates come from differentiating it on the
        # source grid. The motion is resampled afterwards anyway, so nothing more careful is
        # warranted.
        motor_rates = np.gradient(motors, axis=0) * motion["fps"]
        for slot, name in enumerate(mechanism.motor_names):
            positions[name] = motors[:, slot]
            velocities[name] = motor_rates[:, slot]

    joint_pos = np.stack([positions[name] for name in deploy_config.REFERENCE_JOINT_NAMES], axis=-1)
    joint_vel = np.stack([velocities[name] for name in deploy_config.REFERENCE_JOINT_NAMES], axis=-1)
    assert joint_pos.shape == (frames, deploy_config.ACTION_DIM)
    unclipped_joint_pos = joint_pos
    joint_pos = _check_joint_travel(joint_pos)
    clipped_columns = np.any(joint_pos != unclipped_joint_pos, axis=0)
    for column in np.flatnonzero(clipped_columns):
        joint_vel[:, column] = np.gradient(joint_pos[:, column]) * motion["fps"]
    return joint_pos, joint_vel


def _check_joint_travel(joint_pos: np.ndarray) -> np.ndarray:
    """Refuse large travel violations and clamp accepted misses to the hard stops.

    The closed-form linkage inverse can ask for a motor just beyond its stop when the retargeted
    passive pose is slightly outside the coupled workspace. Training accepts a bounded residual
    and PhysX settles at the stop, so this path mirrors that behavior. Publishing the out-of-range
    value itself would leave the controller to clamp silently, hence the explicit clip here.
    """
    limits = np.asarray(deploy_config.REFERENCE_JOINT_LIMITS)
    lower = limits[:, 0] - deploy_config.JOINT_LIMIT_TOLERANCE
    upper = limits[:, 1] + deploy_config.JOINT_LIMIT_TOLERANCE
    violations = (joint_pos < lower) | (joint_pos > upper)
    if violations.any():
        frame, column = (int(index) for index in np.argwhere(violations)[0])
        name = deploy_config.REFERENCE_JOINT_NAMES[column]
        raise ValueError(
            f"Reference joint {name} is {joint_pos[frame, column]:+.4f} rad at frame {frame}, outside the "
            f"robot's travel [{limits[column, 0]:+.4f}, {limits[column, 1]:+.4f}] by more than the "
            f"{deploy_config.JOINT_LIMIT_TOLERANCE:.4f} rad tolerance."
        )

    clipped = np.clip(joint_pos, limits[:, 0], limits[:, 1])
    clipped_mask = clipped != joint_pos
    if clipped_mask.any():
        frame_count = np.unique(np.argwhere(clipped_mask)[:, 0]).size
        worst = float(np.max(np.abs(clipped - joint_pos)))
        print(
            f"[WARN] Clamped {int(clipped_mask.sum())} reference values across {frame_count} frames "
            f"to motor travel limits; worst correction {worst:.4f} rad."
        )
    return clipped


def _lerp(start: np.ndarray, end: np.ndarray, blend: np.ndarray) -> np.ndarray:
    blend = blend.reshape(-1, *([1] * (start.ndim - 1)))
    return start * (1.0 - blend) + end * blend


def _slerp(start: np.ndarray, end: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Shortest-arc slerp over arrays of wxyz quaternions."""
    blend = blend.reshape(-1, *([1] * (start.ndim - 1)))
    dot = (start * end).sum(axis=-1, keepdims=True)
    end = np.where(dot < 0.0, -end, end)
    dot = np.abs(dot).clip(max=1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1.0e-6
    weight_start = np.where(near, 1.0 - blend, np.sin((1.0 - blend) * theta) / np.where(near, 1.0, sin_theta))
    weight_end = np.where(near, blend, np.sin(blend * theta) / np.where(near, 1.0, sin_theta))

    result = weight_start * start + weight_end * end
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _resample_indices(frames: int, source_fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = (frames - 1) / source_fps
    times = np.arange(0.0, duration, 1.0 / deploy_config.REFERENCE_FPS)
    phase = times / duration
    lower = np.floor(phase * (frames - 1)).astype(np.int64)
    upper = np.minimum(lower + 1, frames - 1)
    return lower, upper, phase * (frames - 1) - lower


def build_reference_matrix(motion_path: str | Path) -> np.ndarray:
    """Build the 102-column reference matrix for one retargeted motion."""
    motion = _load_motion(Path(motion_path))
    body_index = motion["body_index"]
    body_states = motion["body_states"]

    joint_pos, joint_vel = _solve_reference_joints(motion)

    base = body_index[deploy_config.BASE_BODY_NAME]
    base_pos = body_states[:, base, 0:3]
    base_quat = body_states[:, base, [6, 3, 4, 5]]
    base_lin_vel_w = body_states[:, base, 7:10]
    base_ang_vel_w = body_states[:, base, 10:13]

    key_ids = [body_index[name] for name in deploy_config.KEY_BODY_NAMES]
    key_pos = body_states[:, key_ids, 0:3]
    key_quat = body_states[:, key_ids][:, :, [6, 3, 4, 5]]

    if abs(motion["fps"] - deploy_config.REFERENCE_FPS) > 1.0e-6:
        lower, upper, blend = _resample_indices(body_states.shape[0], motion["fps"])
        joint_pos = _lerp(joint_pos[lower], joint_pos[upper], blend)
        joint_vel = _lerp(joint_vel[lower], joint_vel[upper], blend)
        base_pos = _lerp(base_pos[lower], base_pos[upper], blend)
        base_lin_vel_w = _lerp(base_lin_vel_w[lower], base_lin_vel_w[upper], blend)
        base_ang_vel_w = _lerp(base_ang_vel_w[lower], base_ang_vel_w[upper], blend)
        key_pos = _lerp(key_pos[lower], key_pos[upper], blend)
        base_quat = _slerp(base_quat[lower], base_quat[upper], blend)
        key_quat = _slerp(key_quat[lower], key_quat[upper], blend)

    yaw_quat = quat_from_yaw(yaw_from_quat_wxyz(base_quat))
    base_pose = np.concatenate([base_pos, base_quat], axis=-1)
    base_vel = np.concatenate(
        [quat_rotate_inverse(yaw_quat, base_lin_vel_w), quat_rotate_inverse(yaw_quat, base_ang_vel_w)],
        axis=-1,
    )

    repeated_base_quat = np.repeat(base_quat[:, None, :], len(key_ids), axis=1)
    key_pos_relative = quat_rotate_inverse(repeated_base_quat, key_pos - base_pos[:, None, :])
    key_quat_relative = quat_mul(quat_conjugate(repeated_base_quat), key_quat)
    key_pose = np.concatenate([key_pos_relative, key_quat_relative], axis=-1).reshape(len(base_pos), -1)

    matrix = np.concatenate([joint_pos, joint_vel, base_pose, base_vel, key_pose], axis=-1)
    _validate_reference_matrix(matrix)
    return matrix.astype(np.float32)


def build_reference_matrix_from_npz(motion_path: str | Path) -> np.ndarray:
    """Build the 102-column reference matrix from the motion.npz used for training."""
    motion = _load_training_motion_npz(Path(motion_path))
    joint_ids = [motion["joint_index"][name] for name in deploy_config.REFERENCE_TRAINING_JOINT_NAMES]
    joint_pos = motion["joint_pos"][:, joint_ids]
    joint_vel = motion["joint_vel"][:, joint_ids]
    unclipped_joint_pos = joint_pos
    joint_pos = _check_joint_travel(joint_pos)
    clipped_columns = np.any(joint_pos != unclipped_joint_pos, axis=0)
    for column in np.flatnonzero(clipped_columns):
        joint_vel[:, column] = np.gradient(joint_pos[:, column]) * motion["fps"]

    body_index = motion["body_index"]
    base = body_index[deploy_config.BASE_BODY_NAME]
    base_pos = motion["body_pos_w"][:, base]
    base_quat = motion["body_quat_w"][:, base]
    base_lin_vel_w = motion["body_lin_vel_w"][:, base]
    base_ang_vel_w = motion["body_ang_vel_w"][:, base]

    key_ids = [body_index[name] for name in deploy_config.KEY_BODY_NAMES]
    key_pos = motion["body_pos_w"][:, key_ids]
    key_quat = motion["body_quat_w"][:, key_ids]

    yaw_quat = quat_from_yaw(yaw_from_quat_wxyz(base_quat))
    base_pose = np.concatenate([base_pos, base_quat], axis=-1)
    base_vel = np.concatenate(
        [quat_rotate_inverse(yaw_quat, base_lin_vel_w), quat_rotate_inverse(yaw_quat, base_ang_vel_w)],
        axis=-1,
    )

    repeated_base_quat = np.repeat(base_quat[:, None, :], len(key_ids), axis=1)
    key_pos_relative = quat_rotate_inverse(repeated_base_quat, key_pos - base_pos[:, None, :])
    key_quat_relative = quat_mul(quat_conjugate(repeated_base_quat), key_quat)
    key_pose = np.concatenate([key_pos_relative, key_quat_relative], axis=-1).reshape(len(base_pos), -1)

    matrix = np.concatenate([joint_pos, joint_vel, base_pose, base_vel, key_pose], axis=-1)
    _validate_reference_matrix(matrix)
    return matrix.astype(np.float32)


def _validate_reference_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2 or matrix.shape[1] != deploy_config.REFERENCE_COLUMN_COUNT:
        raise ValueError(f"Reference matrix is {matrix.shape}, expected N x {deploy_config.REFERENCE_COLUMN_COUNT}.")
    if matrix.shape[0] == 0:
        raise ValueError("Reference matrix has no frames.")
    if not np.isfinite(matrix).all():
        raise ValueError("Reference matrix contains values that are not finite.")

    base_quat_norm = np.linalg.norm(matrix[:, 57:61], axis=-1)
    key_quat_norm = np.linalg.norm(matrix[:, 67:].reshape(matrix.shape[0], -1, 7)[:, :, 3:], axis=-1)
    if np.max(np.abs(base_quat_norm - 1.0)) > QUATERNION_TOLERANCE:
        raise ValueError("Reference base orientation is not a unit quaternion.")
    if np.max(np.abs(key_quat_norm - 1.0)) > QUATERNION_TOLERANCE:
        raise ValueError("Reference key-body orientations are not unit quaternions.")


def _validate_written_text(path: Path, matrix: np.ndarray) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != matrix.shape[0]:
        raise ValueError(f"Wrote {len(lines)} lines for {matrix.shape[0]} frames.")
    for number, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) != deploy_config.REFERENCE_COLUMN_COUNT:
            raise ValueError(
                f"Line {number + 1} has {len(tokens)} columns, expected {deploy_config.REFERENCE_COLUMN_COUNT}."
            )
        if any(len(token.rsplit(".", 1)[-1]) != TEXT_DECIMALS or "." not in token for token in tokens):
            raise ValueError(f"Line {number + 1} is not written with {TEXT_DECIMALS} decimals.")
    written = np.loadtxt(path, ndmin=2)
    if written.shape != matrix.shape:
        raise ValueError(f"Read back {written.shape}, expected {matrix.shape}.")
    deviation = float(np.max(np.abs(written - matrix)))
    if deviation > TEXT_ROUNDING_TOLERANCE:
        raise ValueError(f"Written reference differs from the computed one by {deviation:.3e}.")


def _write_reference_txt(matrix: np.ndarray, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_mode = stat.S_IMODE(output_path.stat().st_mode)
    else:
        current_umask = os.umask(0)
        os.umask(current_umask)
        output_mode = 0o666 & ~current_umask

    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".txt", dir=output_path.parent, delete=False
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        np.savetxt(temporary_path, matrix, fmt=f"%.{TEXT_DECIMALS}f", delimiter=" ")
        _validate_written_text(temporary_path, matrix)
        temporary_path.chmod(output_mode)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def convert_npy_to_txt(motion_path: str | Path, output_path: str | Path) -> Path:
    motion_path = Path(motion_path)
    output_path = Path(output_path)
    if output_path.resolve() == motion_path.resolve():
        raise ValueError("Output path must differ from the input motion.")

    return _write_reference_txt(build_reference_matrix(motion_path), output_path)


def convert_npz_to_txt(motion_path: str | Path, output_path: str | Path) -> Path:
    motion_path = Path(motion_path)
    output_path = Path(output_path)
    if output_path.resolve() == motion_path.resolve():
        raise ValueError("Output path must differ from the input motion.")

    return _write_reference_txt(build_reference_matrix_from_npz(motion_path), output_path)
