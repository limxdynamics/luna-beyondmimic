"""Turn a retargeted .npy into the motion.npz the tracking task consumes.

This runs the four steps of the motion pipeline in order - solve the closed-loop DoF, convert,
hold the final pose, cross-check against forward kinematics - and stops at the first one that
fails. Each step runs as a subprocess because two of them launch their own Isaac Sim and exit the
interpreter when they are done.

``train.py`` calls this automatically when ``--motion_file`` is given a .npy, so the usual reason
to run it by hand is to inspect the warnings before committing to a training run.

The warnings matter more than the exit code. The linkage solve reports poses the achilles pairs
cannot reach, which is a property of the reference rather than of the solve, and it is the one
step whose gate you may legitimately want to widen with --max_unreachable_error once you have
looked at how many frames are involved.

.. code-block:: bash

    python scripts/prepare_motion.py --input motion.npy --output motions/motion.npz
"""

import argparse
import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retargeted .npy to motion.npz pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Retargeted .npy motion.")
    parser.add_argument("--output", type=str, required=True, help="Where to write the motion.npz.")
    parser.add_argument(
        "--robot",
        type=str,
        default="hu_l04_parallel",
        choices=["hu_l04_parallel"],
        help="Supported articulation (fixed to the Luna closed-loop model).",
    )
    parser.add_argument(
        "--tail_seconds",
        type=float,
        default=10.0,
        help=(
            "How long to hold the final pose. Deployment repeats the last frame until an operator"
            " switches controller, and this is what makes the policy practise it. 0 disables."
        ),
    )
    parser.add_argument(
        "--max_unreachable_error",
        type=float,
        default=None,
        help=(
            "Passed to solve_parallel_linkage. Raise it only after reading how many frames the"
            " linkage cannot reach and deciding the residual is acceptable."
        ),
    )
    parser.add_argument(
        "--allow_unreachable_motion",
        action="store_true",
        help="Record linkage quality as a warning instead of rejecting residuals above the limit.",
    )
    parser.add_argument("--linkage_report", type=str, default=None, help="Optional linkage quality report JSON output.")
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Source frame range, both inclusive and 1-based. Use it to drop retargeting artifacts.",
    )
    parser.add_argument("--force_tail", action="store_true", help="Hold a final pose that is still moving.")
    parser.add_argument("--skip_verify", action="store_true", help="Skip the forward-kinematics cross-check.")
    return parser.parse_args()


def run(step: str, argv: list[str]) -> int:
    print(f"\n{'=' * 78}\n[{step}] {' '.join(argv[1:])}\n{'=' * 78}", flush=True)
    return subprocess.run([sys.executable, *argv]).returncode


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.input):
        print(f"[ERROR] {args.input} does not exist.")
        return 1
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    scratch_dir = tempfile.TemporaryDirectory(prefix="prepare_motion_")
    scratch = scratch_dir.name
    linkage = os.path.join(scratch, "linkage.npz")
    # The tail is appended last, so conversion writes somewhere else when a tail is wanted.
    converted = os.path.join(scratch, "converted.npz") if args.tail_seconds > 0 else output

    argv = [
        os.path.join(SCRIPT_DIR, "solve_parallel_linkage.py"),
        "--input_file",
        args.input,
        "--output_file",
        linkage,
    ]
    if args.max_unreachable_error is not None:
        argv += ["--max_unreachable_error", str(args.max_unreachable_error)]
    if args.allow_unreachable_motion:
        argv.append("--allow_unreachable_motion")
    if args.linkage_report is not None:
        argv += ["--report_file", args.linkage_report]
    solve_returncode = run("1/4 solve linkage", argv)
    if solve_returncode != 0 or not os.path.isfile(linkage):
        if solve_returncode == 0:
            print("\n[ERROR] The linkage solver exited without writing its required output.")
        print(
            "\n[ERROR] The linkage solve did not pass its own gate. Read the warning above: it says how"
            "\n        many frames ask for a pose the mechanism cannot reach and by how much. Those"
            "\n        frames carry a tracking ceiling no amount of training will close. If the count"
            "\n        and the residual are acceptable, the direct prepare_motion flag is"
            "\n        --max_unreachable_error; export_policy_and_reference exposes"
            "\n        --motion-max-unreachable-error. Otherwise fix the retarget."
        )
        return 1

    argv = [
        os.path.join(SCRIPT_DIR, "npy_to_npz.py"),
        "--input_file",
        args.input,
        "--robot",
        args.robot,
        "--output_file",
        converted,
    ]
    argv += ["--linkage_file", linkage]
    if args.frame_range is not None:
        argv += ["--frame_range", str(args.frame_range[0]), str(args.frame_range[1])]
    if run("2/4 convert", argv) != 0 or not os.path.isfile(converted):
        if not os.path.isfile(converted):
            print("\n[ERROR] Motion conversion did not write its required output.")
        return 1

    if args.tail_seconds > 0:
        argv = [
            os.path.join(SCRIPT_DIR, "append_motion_tail.py"),
            "--input",
            converted,
            "--output",
            output,
            "--tail_seconds",
            str(args.tail_seconds),
        ]
        if args.force_tail:
            argv.append("--force")
        if run("3/4 hold the final pose", argv) != 0 or not os.path.isfile(output):
            if not os.path.isfile(output):
                print("\n[ERROR] Tail preparation did not write its required output.")
            print(
                "\n[ERROR] The motion is still moving at its last frame, so holding it would ask the robot"
                "\n        to stop dead. This is usually a retargeting artifact in the last few frames:"
                "\n        drop them with --frame_range 1 N and re-run. --force_tail accepts the"
                "\n        discontinuity instead, and --tail_seconds 0 skips the hold altogether."
            )
            return 1
    else:
        print("\n[INFO] 3/4 skipped: --tail_seconds 0, the final pose is not held.")

    if args.skip_verify:
        print("\n[WARN] 4/4 skipped: --skip_verify. Nothing has checked the motion against kinematics.")
    else:
        argv = [
            os.path.join(SCRIPT_DIR, "verify_motion_npz.py"),
            "--motion_file",
            output,
            "--robot",
            args.robot,
        ]
        if run("4/4 verify", argv) != 0:
            print(
                "\n[ERROR] The motion does not match the articulation. Do not train on it. This usually"
                "\n        means the joint or body ordering constants in npy_to_npz.py no longer match the"
                "\n        description - re-run scripts/dump_articulation_order.py and update them."
            )
            return 1

    print(f"\n[INFO] Ready to train: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
