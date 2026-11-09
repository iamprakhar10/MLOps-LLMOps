"""Capture one collection of original public GBFS responses; no AWS resources needed."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import ssl
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

import certifi

DISCOVERY_URL = "https://gbfs.citibikenyc.com/gbfs/gbfs.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "data/citibike/raw"
"""
DEFAULT_OUTPUT: where collected files are saved by default.
__file__ is this script; resolve() gives its absolute path.
parents[0] = citibike, parents[1] = mlops_llmops, parents[2] = src,
parents[3] = MLOps-LLMOps (repository root).
The / operator appends data/citibike/raw to that root.
This location stays the same regardless of the terminal's current directory.
"""

MAX_RESPONSE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class FetchedFeed:
    """Store original feed bytes, parsed JSON, and retrieval metadata."""

    source_url: str
    resolved_url: str
    requested_at: str
    received_at: str
    raw: bytes
    document: dict


def utc_now():
    """Return the current time with an explicit UTC timezone."""

    """
    Typical return: datetime(2026, 11, 9, 10, 0, tzinfo=timezone.utc).
    """

    return datetime.now(timezone.utc)


def fetch_json(url):
    """Fetch HTTPS JSON with retries and preserve the original response."""
    if urlparse(url).scheme != "https" or not urlparse(url).netloc:
        raise ValueError("Feed URLs must be absolute HTTPS URLs")
    request = Request(url, headers={"User-Agent": "citibike-mlops-learning/0.1"})

    """TLS verifies the server certificate; certifi supplies trusted certificate authorities."""

    tls_context = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(3):
        requested_at = utc_now().isoformat()
        try:
            with urlopen(request, timeout=20, context=tls_context) as response:
                """
                raw holds JSON response bytes, not compressed data.
                Read one extra byte to detect responses exceeding the size limit.
                """

                raw = response.read(MAX_RESPONSE_BYTES + 1)
                resolved_url = response.geturl()
            received_at = utc_now().isoformat()
            break
        except (URLError, TimeoutError, ConnectionError) as error:
            """Retry HTTP 429 (rate limiting), server errors, and connection failures."""

            if isinstance(error, HTTPError) and error.code != 429 and error.code < 500:
                raise
            if attempt == 2:
                raise

            """Exponential backoff waits one second, then two, before retrying."""

            time.sleep(2 ** attempt)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Feed response exceeds the 20 MiB collection limit")

    """Parse raw JSON bytes into Python objects; document must be a dictionary."""

    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("Expected a JSON object at the feed root")

    """
    Typical return: a FetchedFeed with URLs, request/receipt times,
    raw=b'{"data": {"stations": [{"station_id": "123"}]}}', and
    document={"data": {"stations": [{"station_id": "123"}]}}.
    The example is shortened; actual content depends on the requested feed.
    """

    return FetchedFeed(url, resolved_url, requested_at, received_at, raw, document)




"""
Typical document:
{
    "data": {
        "en": {
            "feeds": [
                {"name": "station_status", "url": "https://.../station_status.json"},
                {"name": "station_information", "url": "https://.../station_information.json"}
            ]
        }
    }
}
"""
def discover_station_feeds(document):
    """Extract the two station-feed HTTPS URLs from GBFS discovery data."""

    """GBFS is the General Bikeshare Feed Specification; discovery lists its feed URLs."""

    data = document.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("Invalid discovery data")

    """Support both English language-keyed listings and flat feed listings."""

    listing = data.get("en", data)
    feeds = listing.get("feeds") if isinstance(listing, dict) else None
    if not isinstance(feeds, list):
        raise ValueError("Discovery feed has no supported feed listing")
    endpoints = {
        feed.get("name"): feed.get("url")
        for feed in feeds if isinstance(feed, dict)
    }
    required = ("station_status", "station_information")
    for name in required:
        url = endpoints.get(name)
        if not isinstance(url, str) or urlparse(url).scheme != "https" or not urlparse(url).netloc:
            raise ValueError(f"Discovery has no valid HTTPS endpoint for {name}")

    """
    Typical return (URLs shortened):
    {"station_status": "https://.../station_status.json",
     "station_information": "https://.../station_information.json"}
    """

    return {name: endpoints[name] for name in required}


def collect_snapshot(output_dir=DEFAULT_OUTPUT):
    """Save one complete raw collection and return its directory."""
    started_at = utc_now()
    discovery = fetch_json(DISCOVERY_URL)
    endpoints = discover_station_feeds(discovery.document)
    feeds = {"discovery": discovery}
    feeds.update({name: fetch_json(url) for name, url in endpoints.items()})
    """
    Afterward, feeds looks roughly like:
    {
        "discovery": FetchedFeed(...),
        "station_status": FetchedFeed(...),
        "station_information": FetchedFeed(...)
    }
    """


    """A UTC timestamp plus a random UUID keeps separate collections from overwriting each other."""

    collection_id = f"{started_at:%Y%m%dT%H%M%S%fZ}_{uuid4().hex}"
    output_dir = Path(output_dir)
    day_dir = output_dir / f"collected_date={started_at:%Y-%m-%d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    destination = day_dir / collection_id

    """The manifest indexes saved files and records their origin, timestamps, and checksums."""

    manifest = {
        "manifest_version": 1,
        "kind": "raw-feed-collection",
        "collection_id": collection_id,
        "collection_started_at": started_at.isoformat(),
        "feeds": {},
    }

    """Stage files in a temporary directory; publish them together using a same-filesystem rename."""

    with tempfile.TemporaryDirectory(prefix=".collect-", dir=day_dir) as staging:
        for name, feed in feeds.items():
            filename = f"{name}.json.gz"

            """Gzip preserves original bytes; mtime=0 removes a changing timestamp from its header."""

            compressed = gzip.compress(feed.raw, mtime=0)
            (Path(staging) / filename).write_bytes(compressed)

            """
            SHA-256 fingerprints the original bytes for integrity checks.
            TTL is the publisher's cache lifetime; source time differs from receipt time.
            """

            metadata = {
                "file": filename,
                "source_url": feed.source_url,
                "resolved_url": feed.resolved_url,
                "requested_at": feed.requested_at,
                "received_at": feed.received_at,
                "source_last_updated": feed.document.get("last_updated"),
                "source_ttl_seconds": feed.document.get("ttl"),
                "gbfs_version": feed.document.get("version"),
                "sha256_uncompressed": hashlib.sha256(feed.raw).hexdigest(),
                "uncompressed_bytes": len(feed.raw),
                "compressed_bytes": len(compressed),
            }
            data = feed.document.get("data")
            stations = data.get("stations") if isinstance(data, dict) else None
            if isinstance(stations, list):
                metadata["station_count"] = len(stations)
            manifest["feeds"][name] = metadata
        manifest["collection_completed_at"] = utc_now().isoformat()
        (Path(staging) / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        Path(staging).rename(destination)

    """
    Typical return: a Path to the completed collection directory, for example
    Path("/project/data/citibike/raw/collected_date=2026-11-09/<timestamp>_<uuid>").
    """

    return destination


def main():
    """Parse the output directory and run one collection from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        destination = collect_snapshot(args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Collection failed: {error}\n")
    print(f"Saved one raw collection: {destination}")
    print(f"Inspect its source timestamps and hashes: {destination / 'manifest.json'}")

    """
    Typical return: None on success; saves a collection and prints its path.
    Command-line failures exit the process instead of returning a value.
    """



if __name__ == "__main__":
    main()







"""
manifest.json is an index and receipt for one data collection. It describes the downloaded files; it doesn’t contain the station records themselves.

It records:
- Files saved: discovery, station status, station information.
- Origin: requested and resolved URLs.
- Times: when we fetched the data and when the source updated it.
- Checksums: SHA-256 fingerprints to verify the saved contents.
- Details: file sizes, feed versions, and station counts.
We create it so we can later answer:
“Where did these files come from, when were they collected, and are their contents still intact?”

It also gives later processing jobs a clear list of files belonging to one completed collection. It supports traceability and validation; it isn’t a training-dataset version by itself.



==================================
The collector’s job is simply to download today’s station information and availability, then preserve them so we can study changes over time.

It does not predict anything yet.

First, why are we collecting this data?

Our eventual question is:
“Will this station have zero bikes available 30 minutes from now?”

To learn that, we need observations over time:
Station A at 10:00 → 10 bikes
Station A at 10:05 →  7 bikes
Station A at 10:10 →  4 bikes
Station A at 10:30 →  0 bikes
A single download only tells us the current situation. Repeated downloads build the history we need.

Now follow one execution of the collector.

1. We start with ONE known URL.
https://gbfs.citibikenyc.com/gbfs/gbfs.json
A URL is the address from which we request something.

This particular address returns an address book for other data feeds. That is why it is called the discovery feed: it tells us where the other data is.

Its response contains entries like these, with URLs shortened here:
{
  "data": {
    "en": {
      "feeds": [
        {
          "name": "station_status",
          "url": "https://.../station_status.json"
        },
        {
          "name": "station_information",
          "url": "https://.../station_information.json"
        }
      ]
    }
  }
}
This response contains addresses, rather than the current bike counts.

2. We read those addresses.

This function:
discover_station_feeds(discovery.document)
looks through that address book and extracts two URLs:
station_status      → address for current availability
station_information → address for station descriptions
This function does not download anything. It only reads the discovery response we already downloaded.

That is what this docstring means:
“Extract the two station-feed HTTPS URLs from GBFS discovery data.”

“GBFS” is just the name of the standard used to organize bike-sharing feeds. “HTTPS” means the addresses use encrypted HTTP connections.

3. We request data from BOTH extracted URLs.

So a successful collection normally downloads three responses:
Download	What it contains	Why we need it
Discovery	Names and addresses of available feeds	Locate the other feeds and preserve where we found them
Station information	Station IDs, names, coordinates, capacity	Know which station a record describes and where it is
Station status	Available bikes/docks, operational flags, timestamps	Observe availability and how it changes


For example, station information might tell us:
Station ID: 123
Name: Broadway station
Capacity: 30
Station status might tell us:
Station ID: 123
Available bikes: 8
Available docks: 20
The shared station ID connects the two records.

4. We group the three downloaded responses in memory.

That is the purpose of feeds:
feeds
├── discovery response
├── station-status response
└── station-information response
This grouping lets the saving loop handle all three the same way. It is not another URL or another download.

For each response, we retain:
- raw: original JSON bytes, which we will archive.
- document: those same contents parsed into a Python dictionary, which the code can inspect.
- Metadata: URLs and download timestamps.
Parsing raw into document does not make another network request.

5. We save the collection on disk.
One uniquely named collection folder/
├── discovery.json.gz
├── station_status.json.gz
├── station_information.json.gz
└── manifest.json
The three .json.gz files contain the original responses, compressed to save space. Decompressing them restores the original JSON bytes.

document is an in-memory convenience; we don’t save a separate duplicate of it.

6. We create the manifest as the collection’s receipt.

manifest.json describes the other three files:
- Where each came from.
- When we requested and received it.
- When the source says it was updated.
- Its size and checksum.
This lets us trace and verify our saved data later.

7. The program exits.

Currently, one execution creates one collection. Running it again creates another folder without overwriting the earlier one.

Later, AWS will run collection every five minutes while your laptop is off. We will turn that growing history into training data.
One known discovery URL
        ↓
Download its address book
        ↓
Extract two station-data URLs
        ↓
Download information + availability
        ↓
Save three responses + one manifest
        ↓
Repeat later to build history
The goal of these files is to preserve what we actually observed, so later training, evaluation, and debugging can use that exact history.
"""
