import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlops_llmops.citibike import collector


STATUS_URL = "https://example.org/station_status.json"
INFORMATION_URL = "https://example.org/station_information.json"
FEEDS = [
    {"name": "station_status", "url": STATUS_URL},
    {"name": "station_information", "url": INFORMATION_URL},
]


def fetched(document, url):
    raw = json.dumps(document, indent=2).encode()
    return collector.FetchedFeed(
        url, url, "2026-11-09T10:00:00+00:00", "2026-11-09T10:00:01+00:00",
        raw, document,
    )


def fake_fetch(url):
    if url == collector.DISCOVERY_URL:
        return fetched({"version": "1.1", "data": {"en": {"feeds": FEEDS}}}, url)
    return fetched({
        "version": "1.1", "last_updated": 1794218400, "ttl": 60,
        "data": {"stations": [{"station_id": "test-station"}]},
    }, url)


class CollectorTests(unittest.TestCase):
    def test_discovery_accepts_language_and_flat_listings(self):
        for data in ({"en": {"feeds": FEEDS}}, {"feeds": FEEDS}):
            with self.subTest(data=data):
                result = collector.discover_station_feeds({"data": data})
                self.assertEqual(result, {
                    "station_status": STATUS_URL,
                    "station_information": INFORMATION_URL,
                })

    def test_discovery_rejects_missing_or_insecure_endpoints(self):
        for feeds in ([], [{"name": "station_status", "url": "http://example.org"}]):
            with self.subTest(feeds=feeds), self.assertRaises(ValueError):
                collector.discover_station_feeds({"data": {"en": {"feeds": feeds}}})

    @patch.object(collector, "fetch_json", side_effect=fake_fetch)
    def test_snapshot_preserves_original_bytes_and_provenance(self, fetch):
        with tempfile.TemporaryDirectory() as directory:
            result = collector.collect_snapshot(directory)
            manifest = json.loads((result / "manifest.json").read_text())
            self.assertEqual(fetch.call_count, 3)
            self.assertEqual(len(manifest["feeds"]), 3)
            for name, metadata in manifest["feeds"].items():
                raw = gzip.decompress((result / metadata["file"]).read_bytes())
                self.assertEqual(raw, fake_fetch(metadata["source_url"]).raw)
                self.assertEqual(metadata["sha256_uncompressed"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(metadata["uncompressed_bytes"], len(raw))
                self.assertNotEqual(metadata["requested_at"], metadata["received_at"])
                if name != "discovery":
                    self.assertEqual(metadata["station_count"], 1)
                    self.assertEqual(metadata["source_last_updated"], 1794218400)

    @patch.object(collector, "fetch_json", side_effect=fake_fetch)
    def test_repeated_collections_never_overwrite_previous_data(self, _fetch):
        with tempfile.TemporaryDirectory() as directory:
            first = collector.collect_snapshot(directory)
            second = collector.collect_snapshot(directory)
            self.assertNotEqual(first, second)
            self.assertTrue((first / "manifest.json").is_file())
            self.assertTrue((second / "manifest.json").is_file())

    @patch.object(collector, "fetch_json")
    def test_failed_fetch_does_not_publish_a_collection(self, fetch):
        fetch.side_effect = [fake_fetch(collector.DISCOVERY_URL), OSError("feed unavailable")]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OSError):
                collector.collect_snapshot(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    @patch.object(collector, "urlopen")
    def test_fetch_rejects_http_before_network_access(self, urlopen):
        with self.assertRaises(ValueError):
            collector.fetch_json("http://example.org/feed.json")
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
