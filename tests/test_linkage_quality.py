import json
import numpy as np
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from linkage_quality import build_linkage_report, write_linkage_report  # noqa: E402


class LinkageQualityReportTest(unittest.TestCase):
    def test_report_contains_summary_and_contiguous_segments(self):
        report = build_linkage_report(
            np.array([0.0, 0.002, 0.003, 0.0, 0.2, 0.15]),
            residual_tolerance=0.001,
            max_unreachable_error=0.1,
            strict_check=False,
        )

        self.assertEqual(report["frame_count"], 6)
        self.assertEqual(report["unreachable_frame_count"], 4)
        self.assertAlmostEqual(report["unreachable_fraction"], 4 / 6)
        self.assertEqual(report["worst_frame"], 4)
        self.assertEqual(
            [(segment["start_frame"], segment["end_frame"]) for segment in report["segments"]],
            [(1, 2), (4, 5)],
        )
        self.assertFalse(report["strict_check"])
        self.assertFalse(report["within_configured_limit"])
        self.assertIsNone(report["strict_check_passed"])
        self.assertEqual(report["quality_status"], "warning")

    def test_empty_report_is_well_formed(self):
        report = build_linkage_report(
            np.array([]),
            residual_tolerance=0.001,
            max_unreachable_error=0.1,
            strict_check=True,
        )

        self.assertEqual(report["frame_count"], 0)
        self.assertIsNone(report["worst_frame"])
        self.assertEqual(report["segments"], [])
        self.assertEqual(report["quality_status"], "pass")
        self.assertTrue(report["strict_check_passed"])

    def test_strict_report_marks_an_over_limit_motion_as_failed(self):
        report = build_linkage_report(
            np.array([0.0, 0.2]),
            residual_tolerance=0.001,
            max_unreachable_error=0.1,
            strict_check=True,
        )

        self.assertEqual(report["quality_status"], "fail")
        self.assertFalse(report["strict_check_passed"])

    def test_report_writer_publishes_valid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "linkage_report.json"
            report = build_linkage_report(
                np.array([0.0, 0.01]),
                residual_tolerance=0.001,
                max_unreachable_error=0.1,
                strict_check=True,
            )

            write_linkage_report(output, report)

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
