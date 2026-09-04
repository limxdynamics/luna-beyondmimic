"""Compatibility entry point for the standalone ONNX exporter."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.offline_onnx_exporter.export_motion_policy import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
