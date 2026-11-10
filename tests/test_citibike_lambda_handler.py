import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mlops_llmops.citibike import lambda_handler


def make_snapshot(root):
    destination = Path(root) / "collected_date=2026-11-10" / "example-collection"
    destination.mkdir(parents=True)
    manifest = {"collection_id": destination.name, "feeds": {}}
    for name in lambda_handler.FEED_NAMES:
        raw = json.dumps({"feed": name}).encode()
        compressed = gzip.compress(raw, mtime=0)
        filename = f"{name}.json.gz"
        (destination / filename).write_bytes(compressed)
        manifest["feeds"][name] = {
            "file": filename,
            "sha256_uncompressed": hashlib.sha256(raw).hexdigest(),
            "uncompressed_bytes": len(raw),
            "compressed_bytes": len(compressed),
        }
    (destination / "manifest.json").write_text(json.dumps(manifest))
    return destination


class UploadSnapshotTests(unittest.TestCase):
    def test_uploads_original_files_then_manifest_under_raw_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = make_snapshot(directory)
            s3 = Mock()
            uri = lambda_handler.upload_snapshot(snapshot, "test-bucket", s3)
            calls = s3.put_object.call_args_list
            self.assertEqual(len(calls), 4)
            prefix = "raw/collected_date=2026-11-10/example-collection"
            filenames = ["discovery.json.gz", "station_status.json.gz",
                         "station_information.json.gz", "manifest.json"]
            for call, filename in zip(calls, filenames):
                args = call.kwargs
                self.assertEqual(args["Bucket"], "test-bucket")
                self.assertEqual(args["Key"], f"{prefix}/{filename}")
                self.assertEqual(args["Body"], (snapshot / filename).read_bytes())
                self.assertEqual(args["IfNoneMatch"], "*")
                self.assertEqual(args["ServerSideEncryption"], "AES256")
                expected_type = "application/json" if filename == "manifest.json" else "application/gzip"
                self.assertEqual(args["ContentType"], expected_type)
            self.assertEqual(uri, f"s3://test-bucket/{prefix}/manifest.json")

    def test_any_data_upload_failure_prevents_manifest_upload(self):
        for failed_index in range(3):
            with self.subTest(failed_index=failed_index), tempfile.TemporaryDirectory() as directory:
                snapshot = make_snapshot(directory)
                s3 = Mock()
                s3.put_object.side_effect = [{}] * failed_index + [RuntimeError("upload failed")]
                with self.assertRaisesRegex(RuntimeError, "upload failed"):
                    lambda_handler.upload_snapshot(snapshot, "test-bucket", s3)
                keys = [call.kwargs["Key"] for call in s3.put_object.call_args_list]
                self.assertFalse(any(key.endswith("/manifest.json") for key in keys))

    def test_changed_data_fails_before_any_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = make_snapshot(directory)
            (snapshot / "station_information.json.gz").write_bytes(gzip.compress(b"changed"))
            s3 = Mock()
            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                lambda_handler.upload_snapshot(snapshot, "test-bucket", s3)
            s3.put_object.assert_not_called()

    def test_missing_file_fails_before_any_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = make_snapshot(directory)
            (snapshot / "station_information.json.gz").unlink()
            s3 = Mock()
            with self.assertRaises(FileNotFoundError):
                lambda_handler.upload_snapshot(snapshot, "test-bucket", s3)
            s3.put_object.assert_not_called()


class LambdaHandlerTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch.object(lambda_handler, "collect_snapshot")
    @patch.object(lambda_handler.boto3, "client")
    def test_missing_bucket_fails_without_network_or_collection(self, client, collect):
        with self.assertRaisesRegex(ValueError, "RAW_BUCKET"):
            lambda_handler.lambda_handler({}, None)
        collect.assert_not_called()
        client.assert_not_called()

    @patch.dict(os.environ, {"RAW_BUCKET": "test-bucket"})
    @patch.object(lambda_handler, "collect_snapshot")
    @patch.object(lambda_handler.boto3, "client")
    def test_handler_returns_cloud_location_and_cleans_temporary_files(self, client, collect):
        collect.side_effect = lambda *, output_dir: make_snapshot(output_dir)
        result = lambda_handler.lambda_handler({}, None)
        temporary_root = collect.call_args.kwargs["output_dir"]
        self.assertTrue(str(temporary_root).startswith("/tmp/citibike-"))
        self.assertFalse(temporary_root.exists())
        self.assertEqual(result["uploaded_files"], 4)
        self.assertEqual(result["collection_id"], "example-collection")
        self.assertEqual(result["manifest_uri"],
                         "s3://test-bucket/raw/collected_date=2026-11-10/example-collection/manifest.json")

    @patch.dict(os.environ, {"RAW_BUCKET": "test-bucket"})
    @patch.object(lambda_handler, "LOGGER")
    @patch.object(lambda_handler, "collect_snapshot")
    @patch.object(lambda_handler.boto3, "client")
    def test_upload_failure_is_raised_and_temporary_files_are_removed(self, client, collect, logger):
        collect.side_effect = lambda *, output_dir: make_snapshot(output_dir)
        client.return_value.put_object.side_effect = RuntimeError("permission denied")
        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            lambda_handler.lambda_handler({}, None)
        self.assertFalse(collect.call_args.kwargs["output_dir"].exists())
        logger.exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
