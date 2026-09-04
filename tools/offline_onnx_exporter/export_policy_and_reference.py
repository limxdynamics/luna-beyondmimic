"""Export the HU_L04 deployment policy ONNX and DanceLibrary reference TXT together."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from .deploy_config import DEFAULT_RUN_PATH
    from .policy_exporter import export_policy
    from .reference_converter import convert_npz_to_txt
except ImportError:
    from deploy_config import DEFAULT_RUN_PATH
    from policy_exporter import export_policy
    from reference_converter import convert_npz_to_txt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an RSL-RL checkpoint to policy.onnx. When a trusted retargeted .npy is supplied,"
            " also prepare it into an intermediate closed-loop .npz and produce the 102-column"
            " DanceLibrary reference TXT, linkage report, and deployment param.yaml template."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL model_*.pt checkpoint.")
    parser.add_argument(
        "--motion",
        type=Path,
        default=None,
        help="Optional trusted retargeted motion .npy. Omit it to export only policy.onnx.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Destination directory for policy.onnx and optional motion outputs.",
    )
    parser.add_argument(
        "--run-path",
        type=str,
        default=DEFAULT_RUN_PATH,
        help="Value recorded as the run_path metadata entry in the ONNX model.",
    )
    parser.add_argument(
        "--motion-tail-seconds",
        type=float,
        default=10.0,
        help="Final-pose hold applied to the intermediate .npz. 0 disables the hold.",
    )
    parser.add_argument(
        "--motion-max-unreachable-error",
        type=float,
        default=None,
        help="Enable strict motion checking with this maximum linkage residual in radians.",
    )
    parser.add_argument(
        "--strict-motion-check",
        action="store_true",
        help=(
            "Reject a motion whose linkage residual exceeds the configured limit. By default the"
            " exporter records and warns about that quality issue but still produces the reference."
        ),
    )
    return parser.parse_args()


def _ensure_distinct_paths(paths: dict[str, Path]) -> None:
    seen: dict[Path, str] = {}
    for label, path in paths.items():
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"{label} path must differ from {seen[resolved]} path: {resolved}")
        seen[resolved] = label


def _prepare_motion(
    source_motion: Path,
    intermediate: Path,
    *,
    tail_seconds: float,
    max_unreachable_error: float | None,
    strict_motion_check: bool,
    report_output: Path,
) -> None:
    if source_motion.suffix != ".npy":
        raise ValueError(f"Motion {source_motion} must be a trusted retargeted .npy file.")
    if not source_motion.is_file():
        raise ValueError(f"No such motion: {source_motion}")

    repository_root = Path(__file__).resolve().parents[2]
    prepare_script = repository_root / "scripts" / "prepare_motion.py"
    if not prepare_script.is_file():
        raise ValueError(
            "The combined exporter must run from a LimxMimic source checkout because preparing the"
            " closed-loop motion requires scripts/prepare_motion.py and Isaac Sim."
        )

    command = [
        sys.executable,
        str(prepare_script),
        "--input",
        str(source_motion),
        "--output",
        str(intermediate),
        "--robot",
        "hu_l04_parallel",
        "--tail_seconds",
        str(tail_seconds),
        "--linkage_report",
        str(report_output),
    ]
    if max_unreachable_error is not None:
        command += ["--max_unreachable_error", str(max_unreachable_error)]
    if not strict_motion_check:
        command.append("--allow_unreachable_motion")
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(
            "motion preparation failed; see the messages above. The outer strict-limit option is"
            " --motion-max-unreachable-error."
        )


def main() -> int:
    args = parse_args()
    try:
        if not args.checkpoint.is_file():
            raise ValueError(f"No such checkpoint: {args.checkpoint}")
        if args.output_path.exists() and not args.output_path.is_dir():
            raise ValueError(f"output_path must be a directory, got existing file: {args.output_path}")
        policy_output = args.output_path / "policy.onnx"
        reference_output = args.output_path / "reference.txt"
        param_source = Path(__file__).with_name("param.yaml")
        param_output = args.output_path / "param.yaml"
        report_output = args.output_path / "linkage_report.json"
        deprecated_policy_output = args.output_path / "policy" / "test" / "policy.onnx"
        deprecated_reference_output = args.output_path / "motion" / "test.txt"
        paths = {"checkpoint": args.checkpoint, "policy output": policy_output}
        if args.motion is not None:
            paths.update(
                {
                    "motion": args.motion,
                    "reference output": reference_output,
                    "linkage report output": report_output,
                }
            )
        _ensure_distinct_paths(paths)
        policy_output = export_policy(args.checkpoint, policy_output, run_path=args.run_path)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] policy.onnx not generated: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] policy.onnx exported to {policy_output}", flush=True)
    if args.motion is None:
        try:
            for stale_output in (
                reference_output,
                param_output,
                report_output,
                deprecated_policy_output,
                deprecated_reference_output,
            ):
                stale_output.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[FAIL] Could not clean stale motion outputs: {exc}", file=sys.stderr)
            return 1
        print(
            "[INFO] No --motion supplied; reference.txt, param.yaml, and linkage_report.json were not generated.",
            flush=True,
        )
        return 0

    strict_motion_check = args.strict_motion_check or args.motion_max_unreachable_error is not None
    try:
        # Never leave outputs from an earlier full conversion beside a newly exported policy.
        for stale_output in (
            reference_output,
            param_output,
            report_output,
            deprecated_policy_output,
            deprecated_reference_output,
        ):
            stale_output.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="limxmimic_export_motion_") as scratch:
            intermediate = Path(scratch) / "motion.npz"
            _prepare_motion(
                args.motion,
                intermediate,
                tail_seconds=args.motion_tail_seconds,
                max_unreachable_error=args.motion_max_unreachable_error,
                strict_motion_check=strict_motion_check,
                report_output=report_output,
            )
            reference_output = convert_npz_to_txt(intermediate, reference_output)
            shutil.copy2(param_source, param_output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] reference.txt not generated: {exc}", file=sys.stderr)
        print(f"[INFO] policy.onnx remains available at {policy_output}")
        return 1

    print(f"[OK] reference.txt exported to {reference_output}", flush=True)
    print(f"[OK] param.yaml copied to {param_output}", flush=True)
    if report_output.is_file():
        print(f"[OK] linkage_report.json written to {report_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
