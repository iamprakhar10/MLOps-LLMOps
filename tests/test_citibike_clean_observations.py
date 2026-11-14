from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from mlops_llmops.citibike.clean_observations import clean_observations
from test_citibike_health_check import BASE, collection, station


def tables(output):
    return (
        pq.read_table(output / "accepted_observations.parquet"),
        pq.read_table(output / "excluded_observations.parquet"),
    )


class CleanObservationsTests(unittest.TestCase):
    def test_operating_zero_bikes_is_accepted_and_times_remain_separate(self):
        with tempfile.TemporaryDirectory() as workspace:
            raw = Path(workspace) / "raw"
            source = collection(raw, status_rows=[station(num_bikes_available=0)])
            output = clean_observations(raw, output_root=Path(workspace) / "cleaned")
            accepted, excluded = tables(output)
            self.assertEqual(accepted.num_rows, 1)
            self.assertEqual(excluded.num_rows, 0)
            self.assertEqual(accepted.schema, excluded.schema)
            row = accepted.to_pylist()[0]
            self.assertEqual(row["num_bikes_available"], 0)
            self.assertEqual(row["last_reported_at"], BASE - timedelta(seconds=30))
            self.assertEqual(row["received_at"], BASE + timedelta(seconds=1))
            self.assertEqual(row["information_received_at"], BASE + timedelta(seconds=1))
            self.assertEqual(row["source_manifest"], "collected_date=2026-11-11/sample/manifest.json")
            self.assertEqual(row["source_manifest_sha256"], hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest())
            self.assertEqual(row["source_row_index"], 0)
            self.assertNotIn("label", accepted.column_names)
            self.assertNotIn("target", accepted.column_names)

    def test_closed_and_stale_rows_are_excluded_with_reasons(self):
        with tempfile.TemporaryDirectory() as workspace:
            raw = Path(workspace) / "raw"
            collection(raw, status_rows=[station(is_renting=0, last_reported=int(BASE.timestamp()) - 3600)])
            output = clean_observations(raw, output_root=Path(workspace) / "cleaned")
            accepted, excluded = tables(output)
            self.assertEqual(accepted.num_rows, 0)
            row = excluded.to_pylist()[0]
            self.assertIn("is_renting_disabled", row["exclusion_reasons"])
            self.assertIn("stale_station_report", row["exclusion_reasons"])
            self.assertEqual(json.loads(row["source_status_json"])["is_renting"], 0)

    def test_invalid_counts_are_not_changed_to_zero(self):
        for bad in (-1, True, "8", 2 ** 64):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as workspace:
                raw = Path(workspace) / "raw"
                collection(raw, status_rows=[station(num_bikes_available=bad)])
                output = clean_observations(raw, output_root=Path(workspace) / "cleaned")
                accepted, excluded = tables(output)
                self.assertEqual(accepted.num_rows, 0)
                row = excluded.to_pylist()[0]
                self.assertIsNone(row["num_bikes_available"])
                self.assertEqual(json.loads(row["source_status_json"])["num_bikes_available"], bad)
                self.assertIn("invalid_num_bikes_available", row["exclusion_reasons"])

    def test_duplicate_status_ids_and_ambiguous_information_are_excluded(self):
        variants = [
            ({"status_rows": [station(), station()]}, "duplicate_station_id_in_collection", 2),
            ({"information_rows": []}, "missing_station_information", 1),
            ({"information_rows": [{"station_id": "123"}, {"station_id": "123"}]}, "ambiguous_station_information", 1),
        ]
        for kwargs, reason, count in variants:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as workspace:
                raw = Path(workspace) / "raw"
                collection(raw, **kwargs)
                output = clean_observations(raw, output_root=Path(workspace) / "cleaned")
                accepted, excluded = tables(output)
                self.assertEqual(accepted.num_rows, 0)
                self.assertEqual(excluded.num_rows, count)
                self.assertTrue(all(reason in row["exclusion_reasons"] for row in excluded.to_pylist()))

    def test_returning_disabled_alone_does_not_exclude_a_renting_station(self):
        with tempfile.TemporaryDirectory() as workspace:
            raw = Path(workspace) / "raw"
            collection(raw, status_rows=[station(is_returning=0)])
            output = clean_observations(raw, output_root=Path(workspace) / "cleaned")
            accepted, excluded = tables(output)
            self.assertEqual(accepted.num_rows, 1)
            self.assertEqual(excluded.num_rows, 0)
            self.assertIn("is_returning_disabled", accepted.to_pylist()[0]["review_flags"])

    def test_unchanged_report_and_counts_across_collections_are_retained(self):
        with tempfile.TemporaryDirectory() as workspace:
            raw = Path(workspace) / "raw"
            for name, at in [("first", BASE), ("second", BASE + timedelta(minutes=5))]:
                collection(raw, name, at=at, status_rows=[station()])
            output = clean_observations(raw, output_root=Path(workspace) / "cleaned")
            accepted, excluded = tables(output)
            rows = accepted.to_pylist()
            self.assertEqual(len(rows), 2)
            self.assertEqual(excluded.num_rows, 0)
            self.assertEqual(rows[0]["last_reported_at"], rows[1]["last_reported_at"])
            self.assertNotEqual(rows[0]["received_at"], rows[1]["received_at"])

    def test_freshness_threshold_is_configurable_at_receipt_time(self):
        with tempfile.TemporaryDirectory() as workspace:
            raw = Path(workspace) / "raw"
            collection(raw, status_rows=[station(last_reported=int(BASE.timestamp()) - 600)])
            output_root = Path(workspace) / "cleaned"
            first = clean_observations(raw, output_root=output_root, max_age_minutes=15)
            second = clean_observations(raw, output_root=output_root, max_age_minutes=5)
            self.assertNotEqual(first, second)
            self.assertEqual(tables(first)[0].num_rows, 1)
            self.assertEqual(tables(second)[1].num_rows, 1)

    def test_future_station_times_and_missing_information_receipt_are_excluded(self):
        with tempfile.TemporaryDirectory() as workspace:
            raw = Path(workspace) / "raw"
            source = collection(raw, status_rows=[station(last_reported=int(BASE.timestamp()) + 600)])
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["feeds"]["station_information"].pop("received_at")
            manifest_path.write_text(json.dumps(manifest))
            output = clean_observations(raw, output_root=Path(workspace) / "cleaned")
            row = tables(output)[1].to_pylist()[0]
            self.assertIn("future_station_report_timestamp", row["exclusion_reasons"])
            self.assertIn("invalid_information_received_at", row["exclusion_reasons"])

    def test_corrupt_collection_is_skipped_and_good_collection_is_preserved(self):
        with tempfile.TemporaryDirectory() as workspace:
            raw = Path(workspace) / "raw"
            collection(raw, "good")
            bad = collection(raw, "bad")
            (bad / "station_status.json.gz").write_bytes(b"damaged")
            before = {p: p.read_bytes() for p in raw.rglob("*") if p.is_file()}
            output = clean_observations(raw, output_root=Path(workspace) / "cleaned")
            report = json.loads((output / "cleaning_report.json").read_text())
            self.assertEqual(report["counts"]["collections_skipped"], 1)
            self.assertEqual(report["counts"]["accepted"], 1)
            self.assertEqual(report["counts"]["observations_total"], 1)
            self.assertEqual(before, {p: p.read_bytes() for p in raw.rglob("*") if p.is_file()})
            bad_source = next(item for item in report["source_collections"] if item["collection_id"] == "bad")
            self.assertTrue(bad_source["file_issues"])
            self.assertTrue((output / "source_manifest.json").is_file())

    def test_no_dataset_is_published_when_every_collection_fails(self):
        with tempfile.TemporaryDirectory() as workspace:
            raw = Path(workspace) / "raw"
            source = collection(raw)
            (source / "manifest.json").unlink()
            output_root = Path(workspace) / "cleaned"
            with self.assertRaisesRegex(ValueError, "No collections passed"):
                clean_observations(raw, output_root=output_root)
            self.assertEqual(list(output_root.iterdir()), [])

    def test_date_selection_and_output_protection(self):
        with tempfile.TemporaryDirectory() as workspace:
            raw = Path(workspace) / "raw"
            collection(raw, "old", at=BASE - timedelta(days=2))
            collection(raw, "recent")
            output = clean_observations(raw, dates=["2026-11-11"], output_root=Path(workspace) / "cleaned")
            self.assertEqual(tables(output)[0].num_rows, 1)
            for bad_output in (raw, raw / "new"):
                with self.assertRaisesRegex(ValueError, "outside"):
                    clean_observations(raw, output_root=bad_output)
            with self.assertRaisesRegex(ValueError, "missing"):
                clean_observations(raw, dates=["2026-11-11", "2026-11-13"], output_root=Path(workspace) / "cleaned")


if __name__ == "__main__":
    unittest.main()
