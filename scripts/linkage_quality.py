"""Build the machine-readable quality report for a parallel-linkage solve."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def build_linkage_report(
    residuals: np.ndarray,
    *,
    residual_tolerance: float,
    max_unreachable_error: float,
    strict_check: bool,
) -> dict:
    """Summarize per-frame passive-joint residuals and contiguous bad segments."""
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    unreachable = residuals > residual_tolerance
    bad_indices = np.flatnonzero(unreachable)
    segments = []
    if bad_indices.size:
        segment_start = int(bad_indices[0])
        previous = segment_start
        for current_value in bad_indices[1:]:
            current = int(current_value)
            if current != previous + 1:
                segments.append(_segment_report(residuals, segment_start, previous))
                segment_start = current
            previous = current
        segments.append(_segment_report(residuals, segment_start, previous))

    worst_frame = int(np.argmax(residuals)) if residuals.size else None
    worst_residual = float(residuals[worst_frame]) if worst_frame is not None else 0.0
    within_limit = worst_residual <= max_unreachable_error
    quality_status = "pass" if within_limit else ("fail" if strict_check else "warning")
    return {
        "format_version": 1,
        "frame_index_base": 0,
        "frame_count": int(residuals.size),
        "residual_tolerance_rad": float(residual_tolerance),
        "unreachable_frame_count": int(bad_indices.size),
        "unreachable_fraction": float(bad_indices.size / residuals.size) if residuals.size else 0.0,
        "worst_residual_rad": worst_residual,
        "worst_frame": worst_frame,
        "segments": segments,
        "strict_check": bool(strict_check),
        "max_unreachable_error_rad": float(max_unreachable_error),
        "within_configured_limit": within_limit,
        "strict_check_passed": within_limit if strict_check else None,
        "quality_status": quality_status,
    }


def write_linkage_report(path: str | Path, report: dict) -> Path:
    """Write a report atomically enough for the single-process conversion workflow."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _segment_report(residuals: np.ndarray, start: int, end: int) -> dict:
    values = residuals[start : end + 1]
    local_worst = int(np.argmax(values))
    return {
        "start_frame": start,
        "end_frame": end,
        "frame_count": end - start + 1,
        "worst_frame": start + local_worst,
        "worst_residual_rad": float(values[local_worst]),
    }
