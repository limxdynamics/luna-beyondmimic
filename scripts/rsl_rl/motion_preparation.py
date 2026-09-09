"""Shared trusted-NPY preparation for the RSL-RL train and play entry points."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def add_motion_preparation_args(parser: argparse.ArgumentParser) -> None:
    """Add the one supported motion input and its preparation controls."""
    parser.add_argument(
        "--motion_file",
        type=str,
        default=None,
        help="Trusted retargeted .npy motion. It is prepared into a temporary training .npz first.",
    )
    parser.add_argument(
        "--motion_tail_seconds",
        type=float,
        default=10.0,
        help="Final-pose hold applied to the intermediate .npz. 0 disables the hold.",
    )
    parser.add_argument(
        "--motion_max_unreachable_error",
        type=float,
        default=None,
        help="Passed to the linkage solve while preparing the intermediate .npz.",
    )
    parser.add_argument(
        "--motion_force_tail",
        action="store_true",
        help="Hold a final pose the motion has not settled into. The solved linkage DoF are"
        " differentiated at the source frame rate, so a high source fps can report a moving"
        " final frame from solver noise alone; check the 27 motor speeds before using this.",
    )


def prepare_motion_file(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tempfile.TemporaryDirectory:
    """Prepare ``args.motion_file`` and replace it with the temporary NPZ path."""
    if args.motion_file is None:
        parser.error("--motion_file is required")

    source = Path(args.motion_file)
    if source.suffix != ".npy":
        parser.error("--motion_file must be a trusted retargeted .npy; .npz is an internal intermediate")
    if not source.is_file():
        parser.error(f"--motion_file does not exist: {source}")

    scratch = tempfile.TemporaryDirectory(prefix="limxmimic_motion_")
    prepared = Path(scratch.name) / "motion.npz"
    prepare_script = Path(__file__).resolve().parents[1] / "prepare_motion.py"
    command = [
        sys.executable,
        str(prepare_script),
        "--input",
        str(source),
        "--output",
        str(prepared),
        "--robot",
        "hu_l04_parallel",
        "--tail_seconds",
        str(args.motion_tail_seconds),
    ]
    if args.motion_max_unreachable_error is not None:
        command += ["--max_unreachable_error", str(args.motion_max_unreachable_error)]
    if args.motion_force_tail:
        command.append("--force_tail")

    print(f"[INFO] Preparing trusted retargeted motion {source} for hu_l04_parallel.")
    returncode = subprocess.run(command).returncode
    if returncode != 0 or not prepared.is_file():
        scratch.cleanup()
        if returncode == 0:
            raise SystemExit("[ERROR] Motion preparation exited without writing its required .npz output.")
        raise SystemExit("[ERROR] Motion preparation failed; see the message above.")

    args.motion_file = str(prepared)
    return scratch
