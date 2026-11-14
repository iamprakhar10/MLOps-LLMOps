"""Build accepted and excluded observation tables; keep raw files and timestamps unchanged."""

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from mlops_llmops.citibike.collector import DEFAULT_OUTPUT
from mlops_llmops.citibike.health_check import inspect_collection, is_count, parse_timestamp

RULE_VERSION = "observations-v1"
COUNT_FIELDS = ("num_bikes_available", "num_docks_available", "num_bikes_disabled",
                "num_docks_disabled", "num_ebikes_available", "capacity")
TIME_FIELDS = ("collection_started_at", "collection_completed_at", "received_at",
               "information_received_at", "last_reported_at", "status_feed_updated_at",
               "information_feed_updated_at")

"""
A schema states the type of each table column. Keep every timestamp in UTC,
including both source report time and receipt time; neither replaces the other.
Both files use the same schema, even when one contains zero rows.
"""

OBSERVATION_SCHEMA = pa.schema([
    pa.field("collection_id", pa.string()),
    pa.field("source_manifest", pa.string()),
    pa.field("source_manifest_sha256", pa.string()),
    pa.field("source_row_index", pa.int64()),
    pa.field("station_id", pa.string()),
    pa.field("station_name", pa.string()),
    *[pa.field(name, pa.timestamp("us", tz="UTC")) for name in TIME_FIELDS],
    pa.field("report_age_seconds", pa.float64()),
    *[pa.field(name, pa.int64()) for name in COUNT_FIELDS],
    pa.field("is_installed", pa.bool_()),
    pa.field("is_renting", pa.bool_()),
    pa.field("is_returning", pa.bool_()),
    pa.field("review_flags", pa.list_(pa.string())),
    pa.field("exclusion_reasons", pa.list_(pa.string())),
    pa.field("source_status_json", pa.string()),
    pa.field("source_information_json", pa.string()),
], metadata={b"cleaning_rule_version": RULE_VERSION.encode()})


def make_observation(checked, manifest, source_manifest, manifest_hash):
    """Convert one checked station row into table columns and explicit exclusion reasons."""
    observation = checked["observation"]
    information = checked["source_information"] or {}
    status_metadata = manifest["feeds"]["station_status"]
    information_metadata = manifest["feeds"]["station_information"]
    flags = set(observation["flags"])
    times = {
        "collection_started_at": parse_timestamp(manifest.get("collection_started_at")),
        "collection_completed_at": parse_timestamp(manifest.get("collection_completed_at")),
        "received_at": parse_timestamp(status_metadata.get("received_at")),
        "information_received_at": parse_timestamp(information_metadata.get("received_at")),
        "last_reported_at": parse_timestamp(observation.get("last_reported_at")),
        "status_feed_updated_at": parse_timestamp(status_metadata.get("source_last_updated")),
        "information_feed_updated_at": parse_timestamp(information_metadata.get("source_last_updated")),
    }
    for field in ("received_at", "information_received_at"):
        value = times[field]
        if value is None:
            flags.add(f"invalid_{field}")
        elif not times["collection_started_at"] <= value <= times["collection_completed_at"]:
            flags.add(f"{field}_outside_collection")

    """
    Bad values become null in typed columns, with the original row kept as JSON
    in the excluded table. Never turn an invalid count into zero.
    Parquet's signed 64-bit count columns also need a maximum integer bound.
    """

    counts = {}
    for field in COUNT_FIELDS:
        value = information.get(field) if field == "capacity" else (
            checked["source_status"].get(field) if isinstance(checked["source_status"], dict) else None
        )
        counts[field] = value if is_count(value) and value <= 2 ** 63 - 1 else None
        if is_count(value) and value > 2 ** 63 - 1:
            flags.add(f"invalid_{field}")

    """
    A station that refuses returns can still rent bikes, so is_returning_disabled
    alone is a review flag. All other current flags exclude the row under rule v1.
    In particular, zero available bikes at an operating station is accepted.
    """

    reasons = sorted(flags - {"is_returning_disabled"})
    row = {
        "collection_id": manifest["collection_id"],
        "source_manifest": source_manifest,
        "source_manifest_sha256": manifest_hash,
        "source_row_index": observation["row_index"],
        "station_id": observation["station_id"] if isinstance(observation["station_id"], str) else None,
        "station_name": observation["station_name"] if isinstance(observation["station_name"], str) else None,
        **times,
        "report_age_seconds": observation["report_age_seconds"],
        **counts,
        "review_flags": sorted(flags),
        "exclusion_reasons": reasons,
        "source_status_json": json.dumps(checked["source_status"], ensure_ascii=False) if reasons else None,
        "source_information_json": json.dumps(checked["source_information"], ensure_ascii=False) if reasons else None,
    }
    for field in ("is_installed", "is_renting", "is_returning"):
        value = observation[field]
        row[field] = bool(value) if isinstance(value, (bool, int)) and value in (0, 1) else None

    """Typical return: {'station_id': '123', 'num_bikes_available': 0, 'exclusion_reasons': [], ...}."""

    return row


def write_cleaning_summary(summary, directory):
    """Write counts, rules, and skipped collections as JSON and Markdown; return None."""
    (directory / "cleaning_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Citi Bike observation cleaning", "", f"Rule version: {summary['rule_version']}", "",
             "## Counts", "", "| Measure | Value |", "| --- | --- |"]
    lines.extend(f"| {key} | {value} |" for key, value in summary["counts"].items())
    lines += ["", "## Exclusion reasons", "", "| Reason | Rows |", "| --- | --- |"]
    lines.extend(f"| {reason} | {count} |" for reason, count in sorted(summary["exclusion_reason_counts"].items()))
    lines += ["", "## Rules and limits", "",
              f"- Freshness threshold: {summary['settings']['max_age_minutes']} minutes at recorded status receipt time.",
              "- Accepted rows have usable required values, an unambiguous station match, and an installed station allowing rentals.",
              "- Invalid/stale/future reports and duplicate IDs within a collection are excluded.",
              "- Invalid download times, or receipt times outside the collection start/end, are excluded.",
              "- Rentals enabled with zero bikes is accepted. Refusing returns alone does not exclude a station.",
              "- Unchanged counts and repeated station reports across collections are retained.",
              "- Report time, receipt times, feed update times, and collection times remain separate UTC columns.",
              "- Excluded rows retain reasons, source references, and the original row fields as JSON.",
              "- Collections failing file checks contribute no trusted rows; their issues are recorded in cleaning_report.json.",
              "- Reason counts overlap: one excluded row may have several reasons.",
              "- Raw files were not edited. No training labels, gap filling, model training, AWS uploads, or DVC snapshot was created.",
              "- The 15-minute default review rule does not define the future label-matching window."]
    (directory / "cleaning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_observations(raw_root=DEFAULT_OUTPUT, dates=None, output_root=None, max_age_minutes=15):
    """Read local collections and save two Parquet tables plus reports; return the new output path."""
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root or DEFAULT_OUTPUT.parent / "cleaned").resolve()
    if not raw_root.is_dir():
        raise ValueError(f"Raw-data directory does not exist: {raw_root}")
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("Cleaning output must be outside the raw-data directory")
    if not math.isfinite(max_age_minutes) or max_age_minutes <= 0:
        raise ValueError("Maximum observation age must be a positive finite number")
    for value in dates or []:
        date.fromisoformat(value)
    partitions = [raw_root / f"collected_date={value}" for value in sorted(set(dates))] if dates else sorted(raw_root.glob("collected_date=*"))
    if dates and any(not partition.is_dir() for partition in partitions):
        raise ValueError("One or more requested date partitions are missing; check the downloaded dates")
    directories = sorted(
        child for partition in partitions if partition.is_dir()
        for child in partition.iterdir() if child.is_dir() and not child.name.startswith(".")
    )
    if not directories:
        raise ValueError("No collection directories found for the selected dates")
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    destination = output_root / run_id
    counts = Counter(collections_total=len(directories), collections_verified=0,
                     collections_skipped=0, observations_total=0, accepted=0, excluded=0)
    reasons = Counter()
    source_inventory = []
    summary = {
        "rule_version": RULE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "settings": {"dates": sorted(set(dates)) if dates else "all", "max_age_minutes": max_age_minutes,
                     "future_clock_tolerance_seconds": 60},
    }
    output_root.mkdir(parents=True, exist_ok=True)

    """
    Build outputs in a temporary directory, then publish the directory when all files
    are closed. Each run gets its own output folder. Process one collection at a time
    so memory use does not grow with the entire downloaded history.
    """

    with tempfile.TemporaryDirectory(prefix=".clean-", dir=output_root) as temporary_dir:
        staging = Path(temporary_dir)
        with pq.ParquetWriter(staging / "accepted_observations.parquet", OBSERVATION_SCHEMA, compression="zstd") as accepted_writer, \
                pq.ParquetWriter(staging / "excluded_observations.parquet", OBSERVATION_SCHEMA, compression="zstd") as excluded_writer:
            for directory in directories:
                result = inspect_collection(directory, max_age_minutes * 60, include_rows=True)
                manifest_path = directory / "manifest.json"
                source_path = manifest_path.relative_to(raw_root).as_posix()
                manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None
                source = {"collection_id": directory.name, "source_manifest": source_path,
                          "source_manifest_sha256": manifest_hash, "status": result["status"],
                          "file_issues": result["file_issues"]}
                source_inventory.append(source)
                if result["status"] != "verified":
                    counts["collections_skipped"] += 1
                    continue
                counts["collections_verified"] += 1
                accepted_rows, excluded_rows = [], []
                for checked in result["stations"]["checked_rows"]:
                    row = make_observation(checked, result["manifest"], source_path, manifest_hash)
                    if row["exclusion_reasons"]:
                        excluded_rows.append(row)
                        reasons.update(row["exclusion_reasons"])
                    else:
                        accepted_rows.append(row)
                source.update(accepted=len(accepted_rows), excluded=len(excluded_rows))
                counts["observations_total"] += len(accepted_rows) + len(excluded_rows)
                counts["accepted"] += len(accepted_rows)
                counts["excluded"] += len(excluded_rows)
                if accepted_rows:
                    accepted_writer.write_table(pa.Table.from_pylist(accepted_rows, schema=OBSERVATION_SCHEMA))
                if excluded_rows:
                    excluded_writer.write_table(pa.Table.from_pylist(excluded_rows, schema=OBSERVATION_SCHEMA))
        if counts["collections_verified"] == 0:
            raise ValueError("No collections passed the file checks; no cleaned dataset was published")
        summary.update(counts=dict(counts), exclusion_reason_counts=dict(reasons),
                       source_collections=source_inventory)
        write_cleaning_summary(summary, staging)
        (staging / "source_manifest.json").write_text(json.dumps(source_inventory, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)

    """Typical return: Path('/project/data/citibike/cleaned/<run-id>'), containing two tables and reports."""

    return destination


def main():
    """Read command-line options, build the observation tables, and print their location; return None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--date", action="append", dest="dates", help="UTC date YYYY-MM-DD; repeat to select several")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT.parent / "cleaned")
    parser.add_argument("--max-age-minutes", type=float, default=15)
    args = parser.parse_args()
    try:
        destination = clean_observations(args.raw_dir, args.dates, args.output_dir, args.max_age_minutes)
    except (OSError, ValueError, pa.ArrowException) as error:
        parser.exit(1, f"Observation cleaning failed: {error}\n")
    summary = json.loads((destination / "cleaning_report.json").read_text())
    print(json.dumps(summary["counts"], indent=2))
    print(f"Saved tables and reports: {destination}")


if __name__ == "__main__":
    main()
