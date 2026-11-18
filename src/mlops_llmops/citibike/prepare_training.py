"""Build local training examples from cleaned station history; do not fit a model."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mlops_llmops.citibike.clean_observations import OBSERVATION_SCHEMA
from mlops_llmops.citibike.collector import DEFAULT_OUTPUT


CONTRACT_VERSION = "training-examples-v1"
HORIZON = timedelta(minutes=30)
HISTORY_LAG = timedelta(minutes=15)
NEW_YORK = ZoneInfo("America/New_York")
UTC_TIME = pa.timestamp("us", tz="UTC")

"""
Only these columns may enter the future model. The other output columns explain
where an example came from or contain its future answer; they are NOT inputs.
"""

FEATURE_COLUMNS = [
    "station_id", "num_bikes_available", "num_docks_available", "capacity",
    "is_returning", "report_age_seconds", "hour_of_day", "day_of_week",
    "bikes_known_15min_ago", "lag_report_age_seconds", "bike_change_15min",
]
TARGET_COLUMN = "empty_after_30min"
INPUT_COLUMNS = [
    "station_id", "collection_id", "collection_started_at", "collection_completed_at",
    "received_at", "information_received_at", "last_reported_at",
    "num_bikes_available", "num_docks_available", "capacity",
    "is_installed", "is_renting", "is_returning", "exclusion_reasons",
]
TRAINING_SCHEMA = pa.schema([
    pa.field("station_id", pa.string()),
    *[pa.field(name, pa.int64()) for name in (
        "num_bikes_available", "num_docks_available", "capacity",
        "bikes_known_15min_ago", "bike_change_15min",
    )],
    pa.field("is_returning", pa.bool_()),
    pa.field("report_age_seconds", pa.float64()),
    pa.field("lag_report_age_seconds", pa.float64()),
    pa.field("hour_of_day", pa.int8()),
    pa.field("day_of_week", pa.int8()),
    pa.field(TARGET_COLUMN, pa.int8()),
    *[pa.field(name, UTC_TIME) for name in (
        "prediction_at", "current_reported_at", "lag_available_at", "lag_reported_at",
        "target_at", "label_reported_at", "label_available_at",
    )],
    *[pa.field(name, pa.string()) for name in (
        "current_collection_id", "lag_collection_id", "label_collection_id",
    )],
    pa.field("label_num_bikes_available", pa.int64()),
    pa.field("label_time_error_seconds", pa.float64()),
], metadata={b"training_contract_version": CONTRACT_VERSION.encode()})
SKIPPED_SCHEMA = pa.schema([
    pa.field("station_id", pa.string()),
    pa.field("current_collection_id", pa.string()),
    pa.field("prediction_at", UTC_TIME),
    pa.field("target_at", UTC_TIME),
    pa.field("reason", pa.string()),
])


def file_sha256(path: Path) -> str:
    """Read a file in small chunks and return SHA-256 text identifying its bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_history(
    cleaned_dir: Path, station_limit: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Read one cleaned run and return station histories plus input counts and dates.

    Choose up to station_limit IDs, sorted as text, from the earliest accepted
    batch. Future labels do not influence this choice. Also read excluded rows
    for these stations so a known closed/bad observation cannot be silently
    replaced with an older good row when constructing history or future answers.
    """
    paths = {name: cleaned_dir / f"{name}_observations.parquet" for name in ("accepted", "excluded")}
    for path in paths.values():
        schema = pq.ParquetFile(path).schema_arrow
        for name in INPUT_COLUMNS:
            if name not in schema.names or schema.field(name).type != OBSERVATION_SCHEMA.field(name).type:
                raise ValueError(f"Unexpected or missing cleaned column {name!r} in {path.name}")

    header = pq.read_table(paths["accepted"], columns=["station_id", "collection_completed_at"])
    if not header.num_rows or header["collection_completed_at"].null_count:
        raise ValueError("Accepted observations must contain rows with collection completion times")
    first = pc.min(header["collection_completed_at"]).as_py()
    first_ids = header.filter(pc.equal(header["collection_completed_at"], first))["station_id"].to_pylist()
    if any(not isinstance(value, str) or not value for value in first_ids):
        raise ValueError("The first accepted batch contains an invalid station ID")
    cohort = sorted(set(first_ids))[:station_limit]
    coverage_end = pc.max(header["collection_completed_at"]).as_py()
    excluded_times = pq.read_table(paths["excluded"], columns=["collection_completed_at"])
    excluded_end = pc.max(excluded_times["collection_completed_at"]).as_py()
    if excluded_end is not None:
        coverage_end = max(coverage_end, excluded_end)
    grouped = defaultdict(list)
    counts = {}
    for name, path in paths.items():
        table = pq.read_table(path, columns=INPUT_COLUMNS, filters=[("station_id", "in", cohort)])
        counts[f"cohort_{name}_observations"] = table.num_rows
        for row in table.to_pylist():
            if not isinstance(row["collection_completed_at"], datetime):
                raise ValueError("Cannot place a cleaned row in time: collection_completed_at is missing")
            if row["exclusion_reasons"] is None:
                raise ValueError("Cleaned rows must have an explicit exclusion_reasons list")
            if name == "accepted" and row["exclusion_reasons"]:
                raise ValueError("Accepted file contains a row marked as excluded")
            if name == "excluded" and not row["exclusion_reasons"]:
                raise ValueError("Excluded file contains a row without exclusion reasons")
            grouped[row["station_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["collection_completed_at"], row["collection_id"]))
        candidate_times = [row["collection_completed_at"] for row in rows if not row["exclusion_reasons"]]
        if len(candidate_times) != len(set(candidate_times)):
            raise ValueError("A station has multiple accepted rows at the same prediction time")
    return dict(grouped), {
        "cohort_station_ids": cohort, "cohort_selected_at": first.isoformat(),
        "coverage_end_at": coverage_end.isoformat(), "counts": counts,
    }


def observation_issue(row: dict[str, Any], at: datetime, max_age_seconds: float) -> str | None:
    """Return why a row cannot be used at a given time, or None when it is usable.

    Recheck availability and age at this time, not just at the original download.
    For example, a row accepted by the cleaner can be too old 15 minutes later.
    """
    if row["exclusion_reasons"]:
        return "source_excluded"
    for name in ("num_bikes_available", "num_docks_available", "capacity"):
        value = row[name]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 2 ** 63 - 1:
            return "invalid_count"
    if row["is_installed"] is not True or row["is_renting"] is not True:
        return "station_unavailable"
    if not isinstance(row["is_returning"], bool):
        return "invalid_operating_flag"
    names = ("collection_started_at", "collection_completed_at", "received_at",
             "information_received_at", "last_reported_at")
    if any(not isinstance(row[name], datetime) for name in names):
        return "invalid_timestamp"
    start, end = row["collection_started_at"], row["collection_completed_at"]
    if not start <= row["received_at"] <= end or not start <= row["information_received_at"] <= end:
        return "invalid_collection_times"
    if end > at:
        return "inputs_not_available"
    age = (at - row["last_reported_at"]).total_seconds()
    if age < 0:
        return "future_station_report"
    if age > max_age_seconds:
        return "stale_at_use_time"
    return None


def make_features(
    current: dict[str, Any], previous: dict[str, Any] | None, max_age_seconds: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return model inputs and tracking fields, or a reason to skip this example.

    current is the newly completed batch. previous is the latest row already
    available 15 minutes earlier, including a bad row if that was the latest one.
    Future observations are never passed to this function.
    """
    at = current["collection_completed_at"]
    issue = observation_issue(current, at, max_age_seconds)
    if issue:
        return None, f"current_{issue}"
    if previous is None:
        return None, "missing_history"
    lag_at = at - HISTORY_LAG
    issue = observation_issue(previous, lag_at, max_age_seconds)
    if issue:
        return None, f"history_{issue}"
    if current["last_reported_at"] < previous["last_reported_at"]:
        return None, "report_time_went_backwards"
    local = at.astimezone(NEW_YORK)
    return {
        "station_id": current["station_id"],
        "num_bikes_available": current["num_bikes_available"],
        "num_docks_available": current["num_docks_available"],
        "capacity": current["capacity"], "is_returning": current["is_returning"],
        "report_age_seconds": (at - current["last_reported_at"]).total_seconds(),
        "hour_of_day": local.hour, "day_of_week": local.weekday(),
        "bikes_known_15min_ago": previous["num_bikes_available"],
        "lag_report_age_seconds": (lag_at - previous["last_reported_at"]).total_seconds(),
        "bike_change_15min": current["num_bikes_available"] - previous["num_bikes_available"],
        "prediction_at": at, "current_reported_at": current["last_reported_at"],
        "lag_available_at": previous["collection_completed_at"],
        "lag_reported_at": previous["last_reported_at"], "target_at": at + HORIZON,
        "current_collection_id": current["collection_id"],
        "lag_collection_id": previous["collection_id"],
    }, None


def index_future_reports(
    rows: list[dict[str, Any]], max_age_seconds: float,
) -> tuple[list[datetime], list[tuple[dict[str, Any], str | None]]]:
    """Index station report times for finding future answers; return times and rows.

    The same report can appear in several downloads. Use its first received copy;
    a later stale repeat does not erase an earlier usable report. Conflicting bike
    counts or rental/installation states at the same report time are not trusted.
    """
    reports = defaultdict(list)
    for row in rows:
        if isinstance(row["last_reported_at"], datetime):
            reports[row["last_reported_at"]].append(row)
    times = sorted(reports)
    indexed = []
    for at in times:
        copies = reports[at]
        first = min(copies, key=lambda row: (
            row["received_at"] or row["collection_completed_at"],
            row["collection_completed_at"], row["collection_id"],
        ))
        states = {(row["num_bikes_available"], row["is_installed"], row["is_renting"]) for row in copies}
        issue = "conflicting_future_report" if len(states) > 1 else observation_issue(
            first, first["collection_completed_at"], max_age_seconds,
        )
        indexed.append((first, issue))
    return times, indexed


def match_future_report(
    target_at: datetime, times: list[datetime],
    reports: list[tuple[dict[str, Any], str | None]], tolerance_seconds: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the closest future-target report within tolerance, or a skip reason.

    An equal-distance tie chooses the earlier report. If the nearest report is
    unusable, skip the example instead of searching farther for a convenient answer.
    """
    position = bisect_left(times, target_at)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(times)]
    if not candidates:
        return None, "missing_future_report"
    index = min(candidates, key=lambda value: (abs((times[value] - target_at).total_seconds()), times[value]))
    if abs((times[index] - target_at).total_seconds()) > tolerance_seconds:
        return None, "missing_future_report"
    row, issue = reports[index]
    if issue:
        return None, issue if issue == "conflicting_future_report" else "unusable_future_report"
    return row, None


def write_training_report(report: dict[str, Any], directory: Path) -> None:
    """Save counts and the exact preparation rules as JSON and readable Markdown; return None."""
    (directory / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Citi Bike training preparation", "", f"Status: {report['status']}",
             f"Contract: {CONTRACT_VERSION}", "", "## Counts", "", "| Measure | Value |", "| --- | --- |"]
    lines.extend(f"| {name} | {count} |" for name, count in report["counts"].items())
    lines += ["", "## Skipped candidates", "", "| Reason | Rows |", "| --- | --- |"]
    lines.extend(f"| {name} | {count} |" for name, count in sorted(report["skip_reasons"].items()))
    lines += ["", "## Recorded rules", ""]
    lines.extend(f"- {rule}" for rule in report["rules"])
    lines += ["", "## Model inputs", "", ", ".join(f"`{name}`" for name in FEATURE_COLUMNS), "",
              f"Target: `{TARGET_COLUMN}`. All other columns are for tracking/inspection, not model inputs.", "",
              "No train/validation/test split, model fitting, DVC upload, or AWS change was performed.",
              "Later time splits must exclude examples whose target window or label availability crosses the next split boundary.", ""]
    (directory / "training_report.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_training_data(
    cleaned_dir: Path | str, output_root: Path | str | None = None,
    station_limit: int = 100, label_tolerance_minutes: float = 2.5,
    max_age_minutes: float = 15,
) -> Path:
    """Read one cleaned run and save training examples, skipped examples, and reports.

    Args:
        cleaned_dir: Folder containing the accepted/excluded Parquet files and
            cleaning reports from ONE cleaning run; do not combine duplicate runs.
        output_root: Parent folder for a new result directory. None uses
            data/citibike/training. Original inputs are read only.
        station_limit: Maximum station count, chosen from the first accepted batch.
        label_tolerance_minutes: Maximum distance from the desired +30-minute
            target time. This is separate from the stale-report age threshold.
        max_age_minutes: Maximum age of current/history inputs when used. Must
            not exceed the upstream cleaner's limit, which already removed rows.

    Returns:
        Path to a new output folder. Read its report before fitting a model;
        insufficient data can produce an empty table with explicit skip reasons.
    """
    if isinstance(station_limit, bool) or not isinstance(station_limit, int) or station_limit <= 0:
        raise ValueError("Station limit must be a positive integer")
    if not math.isfinite(label_tolerance_minutes) or not 0 <= label_tolerance_minutes < 30:
        raise ValueError("Label tolerance must be finite, nonnegative, and less than 30 minutes")
    if not math.isfinite(max_age_minutes) or max_age_minutes <= 0:
        raise ValueError("Maximum input age must be finite and positive")
    cleaned_dir = Path(cleaned_dir).resolve()
    output_root = Path(output_root or DEFAULT_OUTPUT.parent / "training").resolve()
    for protected in (cleaned_dir, DEFAULT_OUTPUT.resolve()):
        if output_root == protected or protected in output_root.parents:
            raise ValueError("Training output must be outside the cleaned input and raw-data directories")
    names = ["accepted_observations.parquet", "excluded_observations.parquet",
             "cleaning_report.json", "source_manifest.json"]
    inputs = {name: {"sha256": file_sha256(cleaned_dir / name), "bytes": (cleaned_dir / name).stat().st_size}
              for name in names}
    cleaning = json.loads((cleaned_dir / "cleaning_report.json").read_text())
    if max_age_minutes > cleaning["settings"]["max_age_minutes"]:
        raise ValueError("Input age limit exceeds the cleaning threshold; rerun cleaning to recover older rows")
    grouped, inventory = read_history(cleaned_dir, station_limit)
    coverage_end = datetime.fromisoformat(inventory["coverage_end_at"])
    settings = {"station_limit": station_limit, "horizon_minutes": 30, "history_lag_minutes": 15,
                "label_tolerance_minutes": label_tolerance_minutes, "max_age_minutes": max_age_minutes,
                "calendar_timezone": "America/New_York", "prediction_clock": "collection_completed_at"}
    tolerance_seconds, max_age_seconds = label_tolerance_minutes * 60, max_age_minutes * 60
    counts = Counter(inventory["counts"], cohort_stations=len(grouped), candidates=0,
                     examples=0, skipped=0, empty_targets=0, nonempty_targets=0)
    skips = Counter()
    rules = [
        "Prediction time is collection completion time: a local replay assumption, not an S3/API availability guarantee.",
        "Station IDs are sorted and selected from the first accepted batch, without future-label selection.",
        "The 15-minute lag uses the latest row already available by prediction time minus 15 minutes.",
        f"Current and historical reports must be no older than {max_age_minutes} minutes at their respective use times.",
        "Future-dated input reports are rejected here, even inside the cleaner's 60-second clock allowance.",
        "An excluded latest history row blocks that history input; older good rows are not substituted.",
        f"Nearest station report to +30 minutes must fall within +/-{label_tolerance_minutes} minutes; ties choose earlier.",
        "The nearest future report must be usable. Do not jump over a closed/bad report to pick a farther good one.",
        "Repeated reports use their first received copy for labeling; conflicting counts/operational states are skipped.",
        "Wait until the recorded collection history extends past the full future matching window.",
        "Missing history, gaps without a matching report, and unusable targets are skipped; nothing is filled with zero.",
        "Hour and weekday use New York time; weekday is Monday=0 through Sunday=6. Saved timestamps stay UTC.",
        "Target 1 means zero rentable bikes in the matched future report; target 0 means a positive count.",
        "The target describes an observation near +30 minutes, not an exact continuous-time outcome.",
    ]
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    destination = output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_root) as temporary:
        staging = Path(temporary)
        with pq.ParquetWriter(staging / "training_examples.parquet", TRAINING_SCHEMA, compression="zstd") as writer, \
                pq.ParquetWriter(staging / "skipped_examples.parquet", SKIPPED_SCHEMA, compression="zstd") as skipped_writer:
            for station_id in inventory["cohort_station_ids"]:
                rows = grouped[station_id]
                available_times = [row["collection_completed_at"] for row in rows]
                report_times, reports = index_future_reports(rows, max_age_seconds)
                examples, skipped = [], []
                for current in rows:
                    if current["exclusion_reasons"]:
                        continue
                    counts["candidates"] += 1
                    at = current["collection_completed_at"]
                    target_at = at + HORIZON

                    """
                    Look backward by availability time, never by row number. A
                    report received after the historical cutoff cannot be a lag
                    input even if its station report time is earlier than cutoff.
                    """

                    previous_index = bisect_right(available_times, at - HISTORY_LAG) - 1
                    previous = rows[previous_index] if previous_index >= 0 else None
                    features, reason = make_features(current, previous, max_age_seconds)
                    label = None
                    if reason is None:
                        if coverage_end < target_at + timedelta(seconds=tolerance_seconds):
                            reason = "future_window_not_complete"
                        else:
                            label, reason = match_future_report(target_at, report_times, reports, tolerance_seconds)
                    if reason is not None:
                        skips[reason] += 1
                        skipped.append({"station_id": station_id, "current_collection_id": current["collection_id"],
                                        "prediction_at": at, "target_at": target_at, "reason": reason})
                        continue
                    target = int(label["num_bikes_available"] == 0)
                    examples.append({**features, TARGET_COLUMN: target,
                                     "label_reported_at": label["last_reported_at"],
                                     "label_available_at": label["collection_completed_at"],
                                     "label_collection_id": label["collection_id"],
                                     "label_num_bikes_available": label["num_bikes_available"],
                                     "label_time_error_seconds": (label["last_reported_at"] - target_at).total_seconds()})
                    counts["empty_targets" if target else "nonempty_targets"] += 1
                counts["examples"] += len(examples)
                counts["skipped"] += len(skipped)
                if examples:
                    writer.write_table(pa.Table.from_pylist(examples, schema=TRAINING_SCHEMA))
                if skipped:
                    skipped_writer.write_table(pa.Table.from_pylist(skipped, schema=SKIPPED_SCHEMA))
        for name, source in inputs.items():
            if file_sha256(cleaned_dir / name) != source["sha256"]:
                raise ValueError("A cleaned input changed during preparation; no dataset was published")
        manifest = {"contract_version": CONTRACT_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                    "cleaned_dir": str(cleaned_dir), "input_files": inputs, "settings": settings,
                    "cohort_station_ids": inventory["cohort_station_ids"], "cohort_selected_at": inventory["cohort_selected_at"],
                    "builder_sha256": file_sha256(Path(__file__)),
                    "feature_columns": FEATURE_COLUMNS, "target_column": TARGET_COLUMN,
                    "audit_columns": [name for name in TRAINING_SCHEMA.names if name not in FEATURE_COLUMNS + [TARGET_COLUMN]],
                    "output_files": {name: {"sha256": file_sha256(staging / name), "bytes": (staging / name).stat().st_size}
                                     for name in ("training_examples.parquet", "skipped_examples.parquet")}}
        (staging / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        report = {"status": "ready_for_review" if counts["examples"] else "insufficient_usable_examples",
                  "contract_version": CONTRACT_VERSION, "counts": dict(counts), "settings": settings,
                  "coverage_end_at": inventory["coverage_end_at"], "skip_reasons": dict(skips), "rules": rules}
        write_training_report(report, staging)
        staging.rename(destination)
    return destination


def main() -> None:
    """Read terminal options, prepare the dataset, and print its counts/location; return None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-dir", type=Path, required=True, help="One completed cleaning output folder")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--station-limit", type=int, default=100)
    parser.add_argument("--label-tolerance-minutes", type=float, default=2.5)
    parser.add_argument("--max-age-minutes", type=float, default=15)
    args = parser.parse_args()
    try:
        destination = prepare_training_data(args.cleaned_dir, args.output_dir, args.station_limit,
                                            args.label_tolerance_minutes, args.max_age_minutes)
    except (OSError, ValueError, KeyError, pa.ArrowException) as error:
        parser.exit(1, f"Training preparation failed: {error}\n")
    report = json.loads((destination / "training_report.json").read_text())
    print(json.dumps({"status": report["status"], **report["counts"]}, indent=2))
    print(f"Saved training examples and report: {destination}")


if __name__ == "__main__":
    main()



"""
The training table has 24 columns.

Model inputs — 11 columns:
station_id
num_bikes_available
num_docks_available
capacity
is_returning
report_age_seconds
hour_of_day
day_of_week
bikes_known_15min_ago
lag_report_age_seconds
bike_change_15min


Target — 1 column:
empty_after_30min
1 means empty; 0 means bikes available in the matched future report.

Tracking and inspection — 12 columns:

prediction_at
current_reported_at
lag_available_at
lag_reported_at
target_at
label_reported_at
label_available_at
current_collection_id
lag_collection_id
label_collection_id
label_num_bikes_available
label_time_error_seconds
Only the first 11 columns should enter the model. The tracking columns include future information, so they must stay outside the inputs.
"""