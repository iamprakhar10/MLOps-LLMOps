from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from mlops_llmops.citibike.clean_observations import OBSERVATION_SCHEMA
from mlops_llmops.citibike.prepare_training import (
    FEATURE_COLUMNS, TARGET_COLUMN, file_sha256, prepare_training_data,
)


BASE = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)


def observation(minute, bikes=8, station_id="A", reported_minute=None, **changes):
    completed = BASE + timedelta(minutes=minute)
    reported = completed - timedelta(minutes=1) if reported_minute is None else BASE + timedelta(minutes=reported_minute)
    collection_id = f"{station_id}_{minute}"
    row = {name: None for name in OBSERVATION_SCHEMA.names}
    row.update(
        station_id=station_id, station_name=f"Station {station_id}",
        collection_id=collection_id, source_manifest=f"{collection_id}/manifest.json",
        source_manifest_sha256="a" * 64, source_row_index=0,
        collection_started_at=completed - timedelta(seconds=4),
        collection_completed_at=completed, received_at=completed - timedelta(seconds=2),
        information_received_at=completed - timedelta(seconds=1),
        last_reported_at=reported, status_feed_updated_at=completed - timedelta(seconds=3),
        information_feed_updated_at=completed - timedelta(seconds=3),
        num_bikes_available=bikes, num_docks_available=20 - bikes,
        num_bikes_disabled=0, num_docks_disabled=0, num_ebikes_available=0, capacity=20,
        is_installed=True, is_renting=True, is_returning=True,
        review_flags=[], exclusion_reasons=[],
        report_age_seconds=(completed - timedelta(seconds=2) - reported).total_seconds(),
    )
    row.update(changes)
    return row


def history(station_id="A"):
    return [observation(minute, bikes=0 if minute >= 30 else 8, station_id=station_id)
            for minute in range(-20, 66, 5)]


def cleaned_run(root, rows):
    directory = Path(root) / "cleaned"
    directory.mkdir()
    for name, excluded in (("accepted", False), ("excluded", True)):
        group = [row for row in rows if bool(row["exclusion_reasons"]) == excluded]
        pq.write_table(pa.Table.from_pylist(group, schema=OBSERVATION_SCHEMA),
                       directory / f"{name}_observations.parquet")
    (directory / "cleaning_report.json").write_text(json.dumps({"settings": {"max_age_minutes": 15}}))
    (directory / "source_manifest.json").write_text("[]\n")
    return directory


def result_at(output, minute=0, skipped=False):
    filename = "skipped_examples.parquet" if skipped else "training_examples.parquet"
    rows = pq.read_table(output / filename).to_pylist()
    return next(row for row in rows if row["prediction_at"] == BASE + timedelta(minutes=minute))


class PrepareTrainingTests(unittest.TestCase):
    def test_builds_traceable_features_and_binary_targets_without_modifying_inputs(self):
        with tempfile.TemporaryDirectory() as workspace:
            source = cleaned_run(workspace, history())
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            output = prepare_training_data(source, Path(workspace) / "training")
            row = result_at(output)
            self.assertEqual(row["num_bikes_available"], 8)
            self.assertEqual(row["bikes_known_15min_ago"], 8)
            self.assertEqual(row["bike_change_15min"], 0)
            self.assertEqual(row["report_age_seconds"], 60)
            self.assertEqual(row["lag_report_age_seconds"], 60)
            self.assertEqual(row["hour_of_day"], 7)
            self.assertEqual(row["day_of_week"], 2)
            self.assertEqual(row[TARGET_COLUMN], 1)
            self.assertEqual(row["label_num_bikes_available"], 0)
            self.assertEqual(row["target_at"], BASE + timedelta(minutes=30))
            self.assertEqual(row["label_reported_at"], BASE + timedelta(minutes=29))
            self.assertEqual(row["label_time_error_seconds"], -60)
            self.assertEqual(row["label_collection_id"], "A_30")
            self.assertEqual(result_at(output, -5)[TARGET_COLUMN], 0)
            self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})
            manifest = json.loads((output / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["feature_columns"], FEATURE_COLUMNS)
            self.assertNotIn(TARGET_COLUMN, FEATURE_COLUMNS)
            self.assertTrue(all(not name.startswith("label_") for name in FEATURE_COLUMNS))
            self.assertNotIn("information_feed_updated_at", FEATURE_COLUMNS)
            for name, details in manifest["output_files"].items():
                self.assertEqual(file_sha256(output / name), details["sha256"])
            report = json.loads((output / "training_report.json").read_text())
            self.assertEqual(report["counts"]["candidates"], report["counts"]["examples"] + report["counts"]["skipped"])
            self.assertEqual(report["counts"]["examples"], report["counts"]["empty_targets"] + report["counts"]["nonempty_targets"])

    def test_history_uses_availability_not_report_time_or_row_offset(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = [row for row in history() if row["collection_completed_at"] != BASE - timedelta(minutes=15)]
            rows.append(observation(-14, bikes=19, reported_minute=-16))
            source = cleaned_run(workspace, rows)
            output = prepare_training_data(source, Path(workspace) / "training")
            row = result_at(output)
            self.assertEqual(row["bikes_known_15min_ago"], 8)
            self.assertEqual(row["lag_available_at"], BASE - timedelta(minutes=20))
            self.assertEqual(row["lag_report_age_seconds"], 360)

    def test_excluded_latest_history_is_not_replaced_by_older_good_history(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = history()
            at = next(row for row in rows if row["collection_completed_at"] == BASE - timedelta(minutes=15))
            at.update(is_renting=False, exclusion_reasons=["is_renting_disabled"])
            source = cleaned_run(workspace, rows)
            output = prepare_training_data(source, Path(workspace) / "training")
            self.assertEqual(result_at(output, skipped=True)["reason"], "history_source_excluded")

    def test_closest_closed_future_report_blocks_farther_valid_report(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = history()
            rows.append(observation(31, bikes=0, reported_minute=30, is_renting=False,
                                    exclusion_reasons=["is_renting_disabled"]))
            source = cleaned_run(workspace, rows)
            output = prepare_training_data(source, Path(workspace) / "training")
            self.assertEqual(result_at(output, skipped=True)["reason"], "unusable_future_report")

    def test_missing_future_is_skipped_and_tolerance_is_configurable(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = [row for row in history() if row["collection_completed_at"] != BASE + timedelta(minutes=30)]
            source = cleaned_run(workspace, rows)
            outputs = Path(workspace) / "training"
            strict = prepare_training_data(source, outputs)
            wider = prepare_training_data(source, outputs, label_tolerance_minutes=4)
            self.assertEqual(result_at(strict, skipped=True)["reason"], "missing_future_report")
            self.assertEqual(result_at(wider)["label_time_error_seconds"], 240)

    def test_future_match_ties_choose_earlier_report(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = [row for row in history() if row["collection_completed_at"] not in (
                BASE + timedelta(minutes=30), BASE + timedelta(minutes=35))]
            rows += [observation(30, bikes=0, reported_minute=29),
                     observation(32, bikes=3, reported_minute=31)]
            source = cleaned_run(workspace, rows)
            output = prepare_training_data(source, Path(workspace) / "training")
            row = result_at(output)
            self.assertEqual(row["label_reported_at"], BASE + timedelta(minutes=29))
            self.assertEqual(row[TARGET_COLUMN], 1)

    def test_later_stale_repeat_does_not_erase_original_valid_future_report(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = history()
            rows.append(observation(50.5, bikes=0, reported_minute=29,
                                    exclusion_reasons=["stale_station_report"]))
            source = cleaned_run(workspace, rows)
            output = prepare_training_data(source, Path(workspace) / "training")
            self.assertEqual(result_at(output)["label_collection_id"], "A_30")

    def test_conflicting_reports_with_same_timestamp_are_not_trusted_as_labels(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = history() + [observation(31, bikes=4, reported_minute=29)]
            source = cleaned_run(workspace, rows)
            output = prepare_training_data(source, Path(workspace) / "training")
            self.assertEqual(result_at(output, skipped=True)["reason"], "conflicting_future_report")

    def test_current_age_is_rechecked_at_prediction_time(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = history()
            at = next(row for row in rows if row["collection_completed_at"] == BASE)
            at["last_reported_at"] = BASE - timedelta(seconds=901)
            source = cleaned_run(workspace, rows)
            output = prepare_training_data(source, Path(workspace) / "training")
            self.assertEqual(result_at(output, skipped=True)["reason"], "current_stale_at_use_time")

    def test_future_dated_current_input_is_rejected_even_inside_cleaner_allowance(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = history()
            at = next(row for row in rows if row["collection_completed_at"] == BASE)
            at["last_reported_at"] = BASE + timedelta(seconds=15)
            source = cleaned_run(workspace, rows)
            output = prepare_training_data(source, Path(workspace) / "training")
            self.assertEqual(result_at(output, skipped=True)["reason"], "current_future_station_report")

    def test_history_gap_is_not_forward_filled_and_tail_waits_for_future_window(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = [observation(-40), *[row for row in history() if row["collection_completed_at"] >= BASE]]
            source = cleaned_run(workspace, rows)
            output = prepare_training_data(source, Path(workspace) / "training")
            self.assertEqual(result_at(output, skipped=True)["reason"], "history_stale_at_use_time")
            self.assertEqual(result_at(output, minute=35, skipped=True)["reason"], "future_window_not_complete")

    def test_cohort_is_chosen_from_initial_batch_and_cannot_use_later_station_arrivals(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = history("B") + history("C")
            rows += [row for row in history("A") if row["collection_completed_at"] >= BASE]
            source = cleaned_run(workspace, list(reversed(rows)))
            output = prepare_training_data(source, Path(workspace) / "training", station_limit=1)
            manifest = json.loads((output / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["cohort_station_ids"], ["B"])
            self.assertEqual(set(pq.read_table(output / "training_examples.parquet")["station_id"].to_pylist()), {"B"})

    def test_empty_result_still_has_typed_tables_and_explains_missing_history(self):
        with tempfile.TemporaryDirectory() as workspace:
            source = cleaned_run(workspace, [observation(0)])
            output = prepare_training_data(source, Path(workspace) / "training")
            self.assertEqual(pq.read_table(output / "training_examples.parquet").num_rows, 0)
            self.assertEqual(result_at(output, skipped=True)["reason"], "missing_history")
            report = json.loads((output / "training_report.json").read_text())
            self.assertEqual(report["status"], "insufficient_usable_examples")

    def test_invalid_settings_and_output_inside_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as workspace:
            source = cleaned_run(workspace, history())
            for kwargs in ({"station_limit": 0}, {"station_limit": True},
                           {"label_tolerance_minutes": -1}, {"label_tolerance_minutes": 30},
                           {"label_tolerance_minutes": float("nan")},
                           {"max_age_minutes": 0}, {"max_age_minutes": 20}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    prepare_training_data(source, Path(workspace) / "training", **kwargs)
            for output in (source, source / "training"):
                with self.assertRaisesRegex(ValueError, "outside"):
                    prepare_training_data(source, output)


if __name__ == "__main__":
    unittest.main()
