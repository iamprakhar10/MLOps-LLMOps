from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlops_llmops.citibike import health_check


BASE = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)


def station(**updates):
    result = {
        "station_id": "123", "num_bikes_available": 8, "num_docks_available": 12,
        "is_installed": 1, "is_renting": 1, "is_returning": 1,
        "last_reported": int(BASE.timestamp()) - 30,
    }
    result.update(updates)
    return result


def collection(root, name="sample", at=BASE, status_rows=None, information_rows=None, version="1.1"):
    directory = Path(root) / f"collected_date={at:%Y-%m-%d}" / name
    directory.mkdir(parents=True)
    if status_rows is None:
        status_rows = [station(last_reported=int(at.timestamp()) - 30)]
    if information_rows is None:
        information_rows = [{"station_id": "123", "name": "Test station", "capacity": 20}]
    manifest = {
        "collection_id": name, "collection_started_at": at.isoformat(),
        "collection_completed_at": (at + timedelta(seconds=2)).isoformat(), "feeds": {},
    }
    for feed_name in health_check.FEED_NAMES:
        rows = status_rows if feed_name == "station_status" else information_rows
        document = {
            "version": version, "last_updated": int(at.timestamp()) - 10, "ttl": 60,
            "data": {"en": {"feeds": []}} if feed_name == "discovery" else {"stations": rows},
        }
        raw = json.dumps(document).encode()
        compressed = gzip.compress(raw, mtime=0)
        filename = f"{feed_name}.json.gz"
        (directory / filename).write_bytes(compressed)
        metadata = {
            "file": filename, "sha256_uncompressed": hashlib.sha256(raw).hexdigest(),
            "uncompressed_bytes": len(raw), "compressed_bytes": len(compressed),
            "gbfs_version": version, "source_last_updated": document["last_updated"],
            "received_at": (at + timedelta(seconds=1)).isoformat(),
        }
        if feed_name != "discovery":
            metadata["station_count"] = len(rows)
        manifest["feeds"][feed_name] = metadata
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


class HealthCheckTests(unittest.TestCase):
    def test_repeated_counts_across_collections_are_not_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            collection(root, "first")
            collection(root, "second", at=BASE + timedelta(minutes=5))
            report = health_check.check_collections(root)
            self.assertEqual(report["summary"]["collections_verified"], 2)
            self.assertEqual(report["summary"]["rows_without_flags"], 2)
            self.assertEqual(report["summary"]["observed_gaps"], 0)
            self.assertEqual(report["station_issue_counts"], {})

    def test_archive_freshness_uses_recorded_receipt_time(self):
        with tempfile.TemporaryDirectory() as root:
            collection(root, at=datetime(2020, 1, 1, tzinfo=timezone.utc))
            report = health_check.check_collections(root)
            self.assertEqual(report["summary"]["rows_without_flags"], 1)

    def test_closed_and_stale_station_is_reviewed_not_a_file_failure(self):
        with tempfile.TemporaryDirectory() as root:
            collection(root, status_rows=[station(is_renting=0, last_reported=int(BASE.timestamp()) - 3600)])
            report = health_check.check_collections(root)
            self.assertEqual(report["summary"]["collections_verified"], 1)
            self.assertEqual(report["summary"]["rows_with_flags"], 1)
            self.assertEqual(report["station_issue_counts"]["is_renting_disabled"], 1)
            self.assertEqual(report["station_issue_counts"]["stale_station_report"], 1)

    def test_missing_manifest_is_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            directory = collection(root)
            (directory / "manifest.json").unlink()
            report = health_check.check_collections(root)
            self.assertEqual(report["summary"]["collections_incomplete"], 1)
            self.assertEqual(report["summary"]["station_rows_inspected"], 0)

    def test_missing_feed_file_is_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            directory = collection(root)
            (directory / "station_status.json.gz").unlink()
            report = health_check.check_collections(root)
            self.assertEqual(report["summary"]["collections_incomplete"], 1)

    def test_changed_valid_gzip_is_reported_and_rows_are_not_trusted(self):
        with tempfile.TemporaryDirectory() as root:
            directory = collection(root)
            (directory / "station_status.json.gz").write_bytes(gzip.compress(b'{"changed":true}'))
            report = health_check.check_collections(root)
            self.assertEqual(report["summary"]["collections_invalid"], 1)
            self.assertEqual(report["summary"]["station_rows_inspected"], 0)
            self.assertIn("checksum", report["collections"][0]["file_issues"][0])

    def test_malformed_gzip_and_manifest_are_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as root:
            first = collection(root, "bad-gzip")
            (first / "station_status.json.gz").write_bytes(b"not gzip")
            second = collection(root, "bad-manifest")
            (second / "manifest.json").write_text("null")
            report = health_check.check_collections(root)
            self.assertEqual(report["summary"]["collections_invalid"], 2)

    def test_duplicate_ids_are_detected_within_one_collection(self):
        with tempfile.TemporaryDirectory() as root:
            collection(root, status_rows=[station(), station()])
            report = health_check.check_collections(root)
            self.assertEqual(report["station_issue_counts"]["duplicate_station_id_in_collection"], 2)

    def test_missing_and_duplicate_information_matches_are_flagged(self):
        for information_rows, expected in [
            ([], "missing_station_information"),
            ([{"station_id": "123"}, {"station_id": "123"}], "ambiguous_station_information"),
        ]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as root:
                collection(root, information_rows=information_rows)
                report = health_check.check_collections(root)
                self.assertEqual(report["station_issue_counts"][expected], 1)

    def test_missing_ids_bad_counts_and_future_timestamps_are_flagged(self):
        with tempfile.TemporaryDirectory() as root:
            collection(root, status_rows=[station(
                station_id="", num_bikes_available=True, num_docks_available=-1,
                last_reported=int(BASE.timestamp()) + 600,
            )])
            report = health_check.check_collections(root)
            for flag in ("missing_or_invalid_station_id", "invalid_num_bikes_available",
                         "invalid_num_docks_available", "future_station_report_timestamp"):
                self.assertEqual(report["station_issue_counts"][flag], 1)

    def test_timing_jitter_is_allowed_but_large_gaps_are_reported(self):
        with tempfile.TemporaryDirectory() as root:
            collection(root, "first")
            collection(root, "second", at=BASE + timedelta(seconds=305))
            collection(root, "third", at=BASE + timedelta(minutes=15))
            report = health_check.check_collections(root)
            self.assertEqual(len(report["gaps"]), 1)
            self.assertEqual(report["gaps"][0]["elapsed_seconds"], 595)
            self.assertEqual(report["gaps"][0]["estimated_unobserved_intervals"], 1)

    def test_date_filter_excludes_old_manual_sample(self):
        with tempfile.TemporaryDirectory() as root:
            collection(root, "manual", at=BASE - timedelta(days=2))
            collection(root, "scheduled")
            report = health_check.check_collections(root, dates=["2026-11-11"])
            self.assertEqual(report["summary"]["collections_total"], 1)
            self.assertEqual(report["summary"]["observed_gaps"], 0)

    def test_unsupported_schema_is_explicitly_reported(self):
        with tempfile.TemporaryDirectory() as root:
            collection(root, version="3.0")
            report = health_check.check_collections(root)
            self.assertEqual(report["summary"]["collections_invalid"], 1)
            self.assertIn("not supported", str(report["collections"][0]["file_issues"]))

    def test_decompression_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as root:
            collection(root)
            with patch.object(health_check, "MAX_RESPONSE_BYTES", 10):
                report = health_check.check_collections(root)
            self.assertEqual(report["summary"]["collections_invalid"], 1)

    def test_report_writing_preserves_raw_bytes_and_bounds_examples(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace) / "raw"
            collection(root, status_rows=[station(num_bikes_available=-1) for _ in range(10)])
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            report = health_check.check_collections(root)
            output = health_check.write_report(report, Path(workspace) / "reports")
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
            self.assertTrue((output / "health_report.md").is_file())
            self.assertEqual(json.loads((output / "health_report.json").read_text()), report)
            self.assertEqual(report["station_issue_counts"]["invalid_num_bikes_available"], 10)
            self.assertEqual(len(report["station_issue_examples"]["invalid_num_bikes_available"]), 3)
            with self.assertRaisesRegex(ValueError, "outside"):
                health_check.write_report(report, root / "reports")

    def test_invalid_settings_and_empty_input_fail_clearly(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "No collection"):
                health_check.check_collections(root)
            for kwargs in ({"interval_seconds": 0}, {"gap_tolerance_seconds": -1},
                           {"max_age_seconds": float("nan")}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    health_check.check_collections(root, **kwargs)

    def test_timestamp_parsing_rejects_ambiguous_and_invalid_values(self):
        for value in (True, None, 0, -1, "2026-11-11T12:00:00", float("nan"), "bad"):
            with self.subTest(value=value):
                self.assertIsNone(health_check.parse_timestamp(value))
        self.assertEqual(health_check.parse_timestamp(BASE.timestamp()), BASE)
        self.assertEqual(health_check.parse_timestamp("2026-11-11T12:00:00Z"), BASE)


if __name__ == "__main__":
    unittest.main()
