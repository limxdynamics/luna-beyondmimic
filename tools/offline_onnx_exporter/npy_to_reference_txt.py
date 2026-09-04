"""Convert one retargeted HU_L04 motion (.npy) to the DanceLibrary reference TXT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .reference_converter import convert_npy_to_txt
except ImportError:
    from reference_converter import convert_npy_to_txt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a retargeted .npy motion into the 102-column DanceLibrary reference TXT for "
            "HU_L04_01. The motion must come from the retargeting pipeline: it is read with pickle."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Retargeted motion .npy.")
    parser.add_argument("--output", type=Path, required=True, help="Destination reference .txt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.output.resolve() == args.input.resolve():
            raise ValueError("Output path must differ from the input motion.")
        output = convert_npy_to_txt(args.input, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[INFO] Exported DanceLibrary reference: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
