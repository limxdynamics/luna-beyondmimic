"""Export the HU_L04 parallel deployment policy from an RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .deploy_config import DEFAULT_RUN_PATH
    from .policy_exporter import export_policy
except ImportError:
    from deploy_config import DEFAULT_RUN_PATH
    from policy_exporter import export_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an RSL-RL checkpoint to the DanceDeploy policy ONNX for HU_L04_01 running the "
            "closed-loop description with a projected-gravity observation."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL model_*.pt checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Destination policy.onnx.")
    parser.add_argument(
        "--run-path",
        type=str,
        default=DEFAULT_RUN_PATH,
        help="Value recorded as the run_path metadata entry.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.output.resolve() == args.checkpoint.resolve():
            raise ValueError("Output path must differ from the checkpoint path.")
        output = export_policy(args.checkpoint, args.output, run_path=args.run_path)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[INFO] Exported deployment policy: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
