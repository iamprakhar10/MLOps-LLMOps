"""Inspect downloaded Citi Bike collections and write reports without changing raw data."""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from uuid import uuid4
import zlib

from mlops_llmops.citibike.collector import DEFAULT_OUTPUT, MAX_RESPONSE_BYTES

FEED_NAMES = ("discovery", "station_status", "station_information")
SUPPORTED_VERSIONS = {"1.0", "1.1", "2.0", "2.1", "2.2", "2.3"}
EXAMPLES_PER_ISSUE = 3


def parse_timestamp(value):
    """Convert an epoch number or timezone-aware ISO string to UTC; return None if invalid."""
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            if not math.isfinite(value) or value <= 0:
                return None
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError, OSError):
        pass

    """Typical return: datetime(..., tzinfo=timezone.utc), or None for a bad/missing timestamp."""

    return None


def is_count(value):
    """Return True for a nonnegative integer count; reject booleans and numeric strings."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def inspect_stations(status_rows, information_rows, received_at, status_updated_at, max_age_seconds):
    """Count station issues and return a few examples; do not delete or fix any records."""
    information = defaultdict(list)
    information_issues = Counter()
    for row in information_rows:
        if not isinstance(row, dict) or not isinstance(row.get("station_id"), str) or not row["station_id"]:
            information_issues["missing_or_invalid_station_id"] += 1
            continue
        information[row["station_id"]].append(row)
    information_issues["duplicate_station_ids"] = sum(len(rows) > 1 for rows in information.values())

    ids = Counter(
        row["station_id"] for row in status_rows
        if isinstance(row, dict) and isinstance(row.get("station_id"), str) and row["station_id"]
    )
    issues = Counter()
    examples = defaultdict(list)
    normal_examples = []
    flagged_rows = 0
    flags_by_feed = set()
    if received_at is None:
        flags_by_feed.add("invalid_status_received_at")
    if status_updated_at is None:
        flags_by_feed.add("invalid_status_feed_timestamp")
    elif received_at is not None:
        age = (received_at - status_updated_at).total_seconds()
        if age > max_age_seconds:
            flags_by_feed.add("stale_status_feed")
        elif age < -60:
            flags_by_feed.add("future_status_feed_timestamp")

    for index, row in enumerate(status_rows):
        flags = set(flags_by_feed)
        if not isinstance(row, dict):
            row = {}
            flags.add("malformed_station_row")
        station_id = row.get("station_id")
        metadata = None
        if not isinstance(station_id, str) or not station_id:
            flags.add("missing_or_invalid_station_id")
        else:
            if ids[station_id] > 1:
                flags.add("duplicate_station_id_in_collection")
            matches = information.get(station_id, [])
            if not matches:
                flags.add("missing_station_information")
            elif len(matches) > 1:
                flags.add("ambiguous_station_information")
            else:
                metadata = matches[0]
                if not is_count(metadata.get("capacity")):
                    flags.add("missing_or_invalid_capacity")

        for field in ("num_bikes_available", "num_docks_available"):
            if not is_count(row.get(field)):
                flags.add(f"invalid_{field}")
        for field in ("num_bikes_disabled", "num_docks_disabled", "num_ebikes_available"):
            if field in row and not is_count(row[field]):
                flags.add(f"invalid_{field}")
        for field in ("is_installed", "is_renting", "is_returning"):
            value = row.get(field)
            if not isinstance(value, (bool, int)) or value not in (0, 1):
                flags.add(f"invalid_{field}")
            elif not value:
                flags.add(f"{field}_disabled")

        """
        Compare a station's report time with receipt time in THIS collection,
        not today's clock. Archived observations should not become stale just by aging.
        The configurable age threshold is our review rule, not a claim about GBFS compliance.
        """

        reported_at = parse_timestamp(row.get("last_reported"))
        report_age = None
        if reported_at is None:
            flags.add("invalid_station_report_timestamp")
        elif received_at is not None:
            report_age = (received_at - reported_at).total_seconds()
            if report_age > max_age_seconds:
                flags.add("stale_station_report")
            elif report_age < -60:
                flags.add("future_station_report_timestamp")

        example = {
            "row_index": index,
            "station_id": station_id,
            "station_name": metadata.get("name") if metadata else None,
            "num_bikes_available": row.get("num_bikes_available"),
            "num_docks_available": row.get("num_docks_available"),
            "capacity": metadata.get("capacity") if metadata else None,
            "is_installed": row.get("is_installed"),
            "is_renting": row.get("is_renting"),
            "is_returning": row.get("is_returning"),
            "received_at": received_at.isoformat() if received_at else None,
            "last_reported_at": reported_at.isoformat() if reported_at else None,
            "report_age_seconds": report_age,
            "flags": sorted(flags),
        }
        if flags:
            flagged_rows += 1
            for flag in sorted(flags):
                issues[flag] += 1
                if len(examples[flag]) < EXAMPLES_PER_ISSUE:
                    examples[flag].append(example)
        elif len(normal_examples) < EXAMPLES_PER_ISSUE:
            normal_examples.append(example)

    """
    Typical return: counts, issue names, and bounded example lists.
    One row can have several flags. Closed stations are expected states to review,
    not proof of a collector failure. Unchanged counts across collections are allowed.
    """

    return {
        "status_rows": len(status_rows),
        "information_rows": len(information_rows),
        "unique_status_station_ids": len(ids),
        "rows_with_flags": flagged_rows,
        "rows_without_flags": len(status_rows) - flagged_rows,
        "issue_counts": dict(issues),
        "information_issue_counts": dict(information_issues),
        "issue_examples": dict(examples),
        "rows_without_flags_examples": normal_examples,
    }


def inspect_collection(directory, max_age_seconds):
    """Read one collection, verify its three files, and return completeness and station findings."""
    result = {
        "collection_id": directory.name,
        "directory": str(directory),
        "date_partition": directory.parent.name,
        "started_at": None,
        "status": "invalid",
        "file_issues": [],
    }
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        result["status"] = "incomplete"
        result["file_issues"].append("missing_manifest")
        return result
    try:
        manifest = json.loads(manifest_path.read_bytes())
        if not isinstance(manifest, dict) or not isinstance(manifest.get("feeds"), dict):
            raise ValueError("Manifest must contain a feeds dictionary")
    except (OSError, ValueError) as error:
        result["file_issues"].append(f"invalid_manifest: {error}")
        return result

    started_at = parse_timestamp(manifest.get("collection_started_at"))
    completed_at = parse_timestamp(manifest.get("collection_completed_at"))
    if started_at is None:
        result["file_issues"].append("invalid_collection_started_at")
    else:
        result["started_at"] = started_at.isoformat()
        if directory.parent.name != f"collected_date={started_at:%Y-%m-%d}":
            result["file_issues"].append("collection_date_partition_mismatch")
    if completed_at is None or (started_at is not None and completed_at < started_at):
        result["file_issues"].append("invalid_collection_completed_at")
    if manifest.get("collection_id") != directory.name:
        result["file_issues"].append("collection_id_mismatch")

    documents = {}
    for name in FEED_NAMES:
        metadata = manifest["feeds"].get(name)
        if not isinstance(metadata, dict) or metadata.get("file") != f"{name}.json.gz":
            result["file_issues"].append(f"{name}: invalid_file_metadata")
            continue
        filename = directory / f"{name}.json.gz"
        if not filename.is_file():
            result["status"] = "incomplete"
            result["file_issues"].append(f"{name}: missing_file")
            continue
        try:
            compressed = filename.read_bytes()

            """Limit decompression to the collector's response limit rather than trusting file metadata."""

            with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
                raw = stream.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("Decompressed response exceeds the collection limit")
            if (
                len(raw) != metadata.get("uncompressed_bytes")
                or len(compressed) != metadata.get("compressed_bytes")
                or hashlib.sha256(raw).hexdigest() != metadata.get("sha256_uncompressed")
            ):
                raise ValueError("Saved contents do not match manifest sizes/checksum")
            document = json.loads(raw)
            if not isinstance(document, dict):
                raise ValueError("Feed root must be a JSON object")
            if document.get("version") != metadata.get("gbfs_version"):
                raise ValueError("Feed version does not match manifest")
            if document.get("last_updated") != metadata.get("source_last_updated"):
                raise ValueError("Feed timestamp does not match manifest")
            if name != "discovery":
                data = document.get("data")
                rows = data.get("stations") if isinstance(data, dict) else None
                if not isinstance(rows, list):
                    raise ValueError("Expected data.stations to be a list")
                if metadata.get("station_count") != len(rows):
                    raise ValueError("Station count does not match manifest")
                if document.get("version") not in SUPPORTED_VERSIONS:
                    raise ValueError("Station schema version is not supported by this checker")
            documents[name] = document
        except (OSError, EOFError, ValueError, zlib.error) as error:
            result["file_issues"].append(f"{name}: {error}")

    if result["file_issues"]:
        return result
    result["status"] = "verified"
    status_metadata = manifest["feeds"]["station_status"]
    result["stations"] = inspect_stations(
        documents["station_status"]["data"]["stations"],
        documents["station_information"]["data"]["stations"],
        parse_timestamp(status_metadata.get("received_at")),
        parse_timestamp(documents["station_status"].get("last_updated")),
        max_age_seconds,
    )

    """Typical return: {'collection_id': '...', 'status': 'verified', 'stations': {...}, ...}."""

    return result


def check_collections(raw_root=DEFAULT_OUTPUT, dates=None, interval_seconds=300,
                      gap_tolerance_seconds=90, max_age_seconds=900):
    """Audit selected local date partitions and return a report dictionary without editing files."""
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise ValueError(f"Raw-data directory does not exist: {raw_root}")
    if (
        not all(math.isfinite(value) for value in (interval_seconds, gap_tolerance_seconds, max_age_seconds))
        or interval_seconds <= 0 or gap_tolerance_seconds < 0 or max_age_seconds <= 0
    ):
        raise ValueError("Interval/max age must be positive and gap tolerance cannot be negative")
    for value in dates or []:
        date.fromisoformat(value)
    partitions = [raw_root / f"collected_date={value}" for value in sorted(set(dates))] if dates else sorted(raw_root.glob("collected_date=*"))
    directories = sorted(
        child for partition in partitions if partition.is_dir()
        for child in partition.iterdir() if child.is_dir() and not child.name.startswith(".")
    )
    if not directories:
        raise ValueError("No collection directories found for the selected dates")
    collections = [inspect_collection(directory, max_age_seconds) for directory in directories]
    observed = sorted(
        (parse_timestamp(item["started_at"]), item["collection_id"])
        for item in collections if item["started_at"] is not None
    )
    gaps = []
    for (previous, previous_id), (current, current_id) in zip(observed, observed[1:]):
        elapsed = (current - previous).total_seconds()
        if elapsed > interval_seconds + gap_tolerance_seconds:
            gaps.append({
                "previous_collection_id": previous_id,
                "next_collection_id": current_id,
                "previous_started_at": previous.isoformat(),
                "next_started_at": current.isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "estimated_unobserved_intervals": max(1, round(elapsed / interval_seconds) - 1),
            })

    """
    A gap is an observation, not proof of a scheduler failure: manual runs and
    missing local downloads can affect it. We cannot assess times before the first
    observed collection or after the last without an expected schedule window.
    """

    issue_counts = Counter()
    information_counts = Counter()
    issue_examples = defaultdict(list)
    normal_examples = []
    total_rows = flagged_rows = 0
    station_counts = []
    for collection in collections:
        station_report = collection.get("stations")
        if station_report is None:
            continue
        total_rows += station_report["status_rows"]
        flagged_rows += station_report["rows_with_flags"]
        station_counts.append(station_report["unique_status_station_ids"])
        issue_counts.update(station_report["issue_counts"])
        information_counts.update(station_report["information_issue_counts"])
        for flag, rows in station_report["issue_examples"].items():
            for row in rows:
                if len(issue_examples[flag]) < EXAMPLES_PER_ISSUE:
                    issue_examples[flag].append({"collection_id": collection["collection_id"], **row})
        for row in station_report["rows_without_flags_examples"]:
            if len(normal_examples) < EXAMPLES_PER_ISSUE:
                normal_examples.append({"collection_id": collection["collection_id"], **row})
    states = Counter(item["status"] for item in collections)
    report = {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root.resolve()),
        "settings": {
            "dates": sorted(set(dates)) if dates else "all",
            "expected_interval_seconds": interval_seconds,
            "gap_tolerance_seconds": gap_tolerance_seconds,
            "max_observation_age_seconds": max_age_seconds,
            "future_clock_tolerance_seconds": 60,
        },
        "summary": {
            "collections_total": len(collections),
            "collections_verified": states["verified"],
            "collections_incomplete": states["incomplete"],
            "collections_invalid": states["invalid"],
            "first_observed_collection_at": observed[0][0].isoformat() if observed else None,
            "last_observed_collection_at": observed[-1][0].isoformat() if observed else None,
            "observed_gaps": len(gaps),
            "station_rows_inspected": total_rows,
            "rows_with_flags": flagged_rows,
            "rows_without_flags": total_rows - flagged_rows,
            "min_unique_stations_per_collection": min(station_counts) if station_counts else None,
            "max_unique_stations_per_collection": max(station_counts) if station_counts else None,
        },
        "station_issue_counts": dict(issue_counts),
        "information_issue_counts": dict(information_counts),
        "station_issue_examples": dict(issue_examples),
        "rows_without_flags_examples": normal_examples,
        "gaps": gaps,
        "collections": [{key: value for key, value in item.items() if key != "stations"} | {
            "station_counts": {key: item["stations"][key] for key in (
                "status_rows", "information_rows", "unique_status_station_ids", "rows_with_flags", "rows_without_flags"
            )} if "stations" in item else None,
            "station_issue_counts": item.get("stations", {}).get("issue_counts", {}),
        } for item in collections],
    }

    """Typical return: a dictionary of counts, observed gaps, bounded examples, and per-collection findings."""

    return report


def write_report(report, output_root):
    """Save JSON details and a readable Markdown summary in a new directory; return that path."""
    output_root = Path(output_root).resolve()
    raw_root = Path(report["raw_root"]).resolve()
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("Report output must be outside the raw-data directory")
    destination = output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "health_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Citi Bike collection health", "", f"Generated: {report['generated_at']}", "",
             "## Summary", "", "| Measure | Value |", "| --- | --- |"]
    lines.extend(f"| {key} | {value} |" for key, value in report["summary"].items())
    lines += ["", "## Station review flags", "", "| Flag | Rows |", "| --- | --- |"]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(report["station_issue_counts"].items()))
    lines += ["", "## Collections needing file checks", ""]
    problems = [item for item in report["collections"] if item["status"] != "verified"]
    lines += [f"- {item['collection_id']}: {'; '.join(item['file_issues'])}" for item in problems] or ["None found."]
    lines += ["", "## Observed timing gaps", ""]
    lines += [f"- {gap['previous_started_at']} to {gap['next_started_at']}: {gap['elapsed_seconds']} seconds"
              for gap in report["gaps"]] or ["None above the configured tolerance."]
    lines += ["", "## How to interpret this report", "",
              "- Verified means the saved files match their manifest and have supported basic structures; it does not certify source accuracy.",
              "- Closed/disabled stations are review flags, not automatically collection failures.",
              "- A row can have several flags; flag totals can exceed the number of flagged rows.",
              "- Freshness is measured at recorded receipt time, not the audit date. The age threshold is configurable.",
              "- Unchanged counts across collections are allowed. Duplicate IDs are checked within each collection only.",
              "- Gaps are between observed collection starts; they do not prove missing scheduled runs or cover the start/end of a schedule.",
              "- Files with integrity/schema issues are excluded from station summaries.",
              "- Raw data was not edited. No cleaning, labeling, model training, or DVC snapshot was performed.",
              "- See health_report.json for settings, per-collection results, and actual station examples."]
    (destination / "health_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    """Typical return: Path('/project/data/citibike/reports/<audit-id>')."""

    return destination


def main():
    """Read command-line settings, audit local downloads, and print report locations; return None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--date", action="append", dest="dates", help="UTC date YYYY-MM-DD; repeat to select several")
    parser.add_argument("--interval-seconds", type=float, default=300)
    parser.add_argument("--gap-tolerance-seconds", type=float, default=90)
    parser.add_argument("--max-age-minutes", type=float, default=15)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT.parent / "reports")
    args = parser.parse_args()
    try:
        report = check_collections(args.raw_dir, args.dates, args.interval_seconds,
                                   args.gap_tolerance_seconds, args.max_age_minutes * 60)
        destination = write_report(report, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Health check failed: {error}\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Readable report: {destination / 'health_report.md'}")
    print(f"Detailed report: {destination / 'health_report.json'}")


if __name__ == "__main__":
    main()
