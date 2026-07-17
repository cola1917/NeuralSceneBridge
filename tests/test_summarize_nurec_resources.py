from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.summarize_nurec_resources import summarize


class SummarizeNuRecResourcesTests(unittest.TestCase):
    def test_reports_peak_capacity_and_minimum_free_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resource_log = Path(directory) / "resources.csv"
            resource_log.write_text(
                "timestamp_utc,gpu_memory_used_mib,gpu_memory_total_mib,"
                "gpu_utilization_percent,power_draw_w,temperature_c,"
                "host_memory_available_kib,disk_available_kib\n"
                "2026-07-17T00:00:00Z,100,32760,10,40,45,50000,90000\n"
                "2026-07-17T00:00:02Z,31000,32760,99,300,70,20000,80000\n",
                encoding="utf-8",
            )

            summary = summarize(resource_log)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["observed_duration_seconds"], 2.0)
        self.assertEqual(summary["gpu_memory_peak_mib"], 31000.0)
        self.assertEqual(summary["gpu_memory_total_mib"], 32760.0)
        self.assertEqual(summary["gpu_utilization_peak_percent"], 99.0)
        self.assertEqual(summary["host_memory_available_min_kib"], 20000.0)
        self.assertEqual(summary["disk_available_min_kib"], 80000.0)


if __name__ == "__main__":
    unittest.main()
