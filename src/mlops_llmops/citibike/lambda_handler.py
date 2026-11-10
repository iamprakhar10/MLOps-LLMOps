"""Download the Citi Bike feeds in Lambda and save the completed files in S3."""

import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

import boto3
from botocore.config import Config

from mlops_llmops.citibike.collector import collect_snapshot

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
FEED_NAMES = ("discovery", "station_status", "station_information")


def upload_snapshot(snapshot_dir, bucket, s3_client):
    """Check the saved files, then send them to the chosen S3 bucket.

    snapshot_dir: folder created by the collector, containing three compressed
        feed files and manifest.json. In Lambda this folder is under /tmp.
    bucket: destination bucket name, such as citibike-raw-prakhar-20261110.
    s3_client: Boto3's S3 helper, used to send upload requests to AWS.

    Read each file and compare its name, byte sizes, and SHA-256 fingerprint
    with the manifest. Stop if a file is missing or does not match.
    Upload the three data files first and manifest.json last, so later jobs
    can treat the manifest's presence as the signal that the files are ready.

    Return the uploaded manifest's address as a string: s3://bucket/raw/.../manifest.json.
    A failed check or upload raises an error instead of reporting success.
    """
    snapshot_dir = Path(snapshot_dir)
    """Read the manifest as bytes, then parse it into a dictionary of saved file details."""

    manifest_bytes = (snapshot_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    """Ensure this manifest describes the current folder, not a different collection."""

    if manifest["collection_id"] != snapshot_dir.name:
        raise ValueError("Manifest collection ID does not match its directory")

    """
    SHA-256 produces a fingerprint from file bytes; hexdigest() turns it into text.
    Compare a newly calculated fingerprint with the one saved during collection.
    This checks that files are unchanged, not that bike counts or timestamps are correct.
    """

    """Prepare a list of (filename, file bytes, file type) entries before uploading anything."""

    uploads = []
    for name in FEED_NAMES:
        metadata = manifest["feeds"][name]
        filename = f"{name}.json.gz"
        """The manifest must name the expected file, such as station_status.json.gz."""

        if metadata["file"] != filename:
            raise ValueError(f"Unexpected filename for {name}")
        compressed = (snapshot_dir / filename).read_bytes()
        """Recover the original JSON bytes in memory to check them; keep gzip bytes for upload."""

        raw = gzip.decompress(compressed)
        """Check the original-byte fingerprint, original byte count, and compressed byte count."""

        if (
            hashlib.sha256(raw).hexdigest() != metadata["sha256_uncompressed"]
            or len(raw) != metadata["uncompressed_bytes"]
            or len(compressed) != metadata["compressed_bytes"]
        ):
            raise ValueError(f"Collection integrity check failed for {name}")
        uploads.append((filename, compressed, "application/gzip"))

    """
    An S3 "key" is the full name of a stored file. A "prefix" is its shared beginning:
    raw/collected_date=YYYY-MM-DD/<collection_id>.
    Add /station_status.json.gz to get one complete key. Our IAM role permits raw/*.
    """

    prefix = f"raw/{snapshot_dir.parent.name}/{snapshot_dir.name}"

    """
    Append the manifest after the three data files so it uploads last.
    If a data upload fails, stop without uploading the manifest.
    Some data files may remain in S3, but later jobs must ignore collections without a manifest.
    """

    uploads.append(("manifest.json", manifest_bytes, "application/json"))
    """
    put_object uploads one file:
    Bucket = destination bucket; Key = complete stored filename; Body = file bytes.
    ContentType describes the bytes: application/gzip or application/json.
    ServerSideEncryption="AES256" asks S3 to encrypt using S3-managed keys.
    IfNoneMatch="*" rejects an existing key instead of replacing its contents.
    """

    for filename, body, content_type in uploads:
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/{filename}",
            Body=body,
            ContentType=content_type,
            ServerSideEncryption="AES256",
            IfNoneMatch="*",
        )

    """
    An s3:// address identifies a bucket and file; it is not a public download link.
    Typical return: "s3://my-bucket/raw/collected_date=2026-11-10/<id>/manifest.json".
    """

    return f"s3://{bucket}/{prefix}/manifest.json"


def lambda_handler(event, context):
    """Start one download-and-upload cycle when AWS calls this Python function.

    event: information from the test or schedule that started this run.
    context: execution details supplied by AWS. Neither argument is used here yet.

    Read RAW_BUCKET from Lambda settings. Run the existing collector inside a
    temporary /tmp folder, upload the completed files to S3, then clean up the folder.
    Log any failure and raise it again so AWS knows that this run failed.

    Return a dictionary with the collection ID, S3 manifest address, and file count.
    """

    """
    RAW_BUCKET is the setting added under Lambda -> Configuration -> Environment variables.
    Read the destination from that setting rather than from an incoming test event.
    """

    bucket = os.environ.get("RAW_BUCKET", "").strip()
    if not bucket:
        raise ValueError("Set the RAW_BUCKET environment variable before invoking Lambda")

    try:
        """
        Create a working folder such as /tmp/citibike-abc123.
        The collector builds its files inside it; S3 holds their permanent copy.
        Leaving this with block removes our working folder, including on ordinary exceptions.
        """

        with tempfile.TemporaryDirectory(prefix="citibike-", dir="/tmp") as temporary_dir:
            """Reuse the existing collector, but save under /tmp instead of its laptop default."""

            snapshot_dir = collect_snapshot(output_dir=Path(temporary_dir))

            """
            Boto3 is the Python library for calling AWS services; client("s3") creates
            its S3 helper. In Lambda, AWS credentials come from the attached IAM role.
            Config allows 5 seconds to connect, 10 seconds for response reads, and
            3 attempts per S3 request in total, including the initial attempt.
            """

            s3_client = boto3.client(
                "s3",
                config=Config(
                    connect_timeout=5,
                    read_timeout=10,
                    retries={"mode": "standard", "total_max_attempts": 3},
                ),
            )
            manifest_uri = upload_snapshot(snapshot_dir, bucket, s3_client)
            """Describe the completed upload: one collection, its manifest address, and four files."""

            result = {
                "collection_id": snapshot_dir.name,
                "manifest_uri": manifest_uri,
                "uploaded_files": len(FEED_NAMES) + 1,
            }
    except Exception:
        """Write error details to CloudWatch logs, then raise again so AWS records failure."""

        LOGGER.exception("Citi Bike collection/upload failed")
        raise

    """Log a successful result; %s is replaced with the result dictionary formatted as JSON."""

    LOGGER.info("Collection completed: %s", json.dumps(result))

    """
    Typical return: {"collection_id": "<id>", "manifest_uri": "s3://.../manifest.json",
                     "uploaded_files": 4}. This is a result dictionary, not an HTTP response.
    """

    return result



"""
Lambda provides writable temporary storage at /tmp.
Create our working directory under /tmp
        ↓
Collector builds a complete collection inside it
        ↓
Upload the collection to S3
        ↓
Remove our working directory



/tmp/citibike-<random>/             ← handler's working directory
└── collected_date=.../
    └── .collect-<random>/         ← collector prepares files here
        ↓ renamed when complete
    └── <collection-id>/          ← uploaded to S3, then cleaned up



A checksum is a code calculated from a file’s contents. We save it so we can later check whether those contents changed.

In our project, that code is a 64-character string.
Follow one file through our code:

1. Download the JSON

Suppose the downloaded contents are:
{"bikes": 8}
The collector holds those original contents in feed.raw.

2. Calculate a code from those contents
hashlib.sha256(feed.raw).hexdigest()
- sha256(...) applies a fixed calculation to the contents.
- .hexdigest() gives us the result as readable letters and numbers.
The calculation is not random. Exactly the same contents always produce exactly the same code.

3. Save the contents and the code separately
Original contents → compress → station_status.json.gz

Calculated code  → record in manifest.json
Compression and calculating the code are independent operations.

4. Read the saved file before uploading

The upload function reads station_status.json.gz and decompresses it. Now it has the saved file’s original JSON contents again.

5. Calculate a code again
hashlib.sha256(raw).hexdigest()
This uses the same calculation, now on the contents recovered from the saved file.

6. Compare the two results
Code recorded during download
              ↕ compare
Code calculated from the saved file
- Same code: no content change detected; continue uploading.
- Different code: the contents differ; stop.
For example, changing {"bikes": 8} to {"bikes": 9} would produce a different code, even though the file size stays the same.

Why check twice? The first calculation records what we downloaded. The second checks what we are about to upload. In our immediate workflow this is a defensive check, rather than a requirement for using S3.
"""