# Citi Bike availability forecasting

Predict the probability that a station will have **zero rentable bikes in
30 minutes**. This is an independent portfolio project using public data.

## Constraints and milestones

- AWS region: Mumbai (`ap-south-1`); Free plan, initially $110 credits and
  140 days remaining as reported on 2026-11-09.
- First cloud milestone spending allocation: up to $10 of credits. This is
  a planning allocation, not an automatic spending cap or a cost estimate.
- Continuous lightweight collection and prediction; training and MLflow compute
  run on demand. Export important data before account access expires.
- Start with a versioned cohort of around 100 stations for modeling. The public
  feed returns the full system, so raw responses will initially include all stations.

| Step | Deliverable | Completion check |
| --- | --- | --- |
| 1 — current | One-shot collector, run locally | Original feed responses and a verifiable manifest are saved |
| 2 | S3 + scheduled Lambda collection | New observations arrive every five minutes with the laptop off |
| 3 | Validation, curated data, DVC + S3 versions | Retrieve an earlier dataset and safely retry processing |
| 4 | Features, delayed labels, time-based evaluation, MLflow | Compare algorithms on the same frozen data and splits |
| 5 | Registered model and cloud prediction API | Trace an API prediction to its features and model release |
| 6 | Monitoring, CI/CD, rollback and teardown | Evaluate later outcomes, recover from failures, preserve evidence |

Only step 1 has been implemented. Nothing has been deployed to AWS.

## Where the data comes from

- [Official Citi Bike system data](https://citibikenyc.com/system-data)
- [GBFS discovery feed](https://gbfs.citibikenyc.com/gbfs/gbfs.json)
- [Citi Bike data use policy](https://citibikenyc.com/data-sharing-policy)

The collector reads the discovery feed to find `station_status` (changing
availability) and `station_information` (station metadata). It does not hard-code
the provider's underlying versioned endpoints.

## Run the first collection

From the repository root, using the existing uv environment:

```bash
uv run python -m mlops_llmops.citibike.collector
```

This fetches one collection and exits. It does not schedule a background job or
create AWS resources. Run it again after five minutes to get a second collection.
Unchanged source values are valid: fetching again does not force the source to update.

The default output is anchored to the repository, regardless of the terminal's
working directory. Override it with `--output-dir /path/to/raw` if needed.

```text
data/citibike/raw/
  collected_date=YYYY-MM-DD/
    <UTC timestamp>_<unique ID>/
      discovery.json.gz
      station_status.json.gz
      station_information.json.gz
      manifest.json
```

Each `.json.gz` contains the original downloaded JSON bytes, compressed without
rewriting the data. `manifest.json` records source/resolved URLs, UTC request and
receive times, source timestamps, feed versions, byte sizes, and SHA-256 hashes of
the decompressed bytes. Station counts are included when available.

Open the manifest first. To inspect a status file, decompress it with a gzip-aware
viewer. The collector publishes a collection directory only after all three
responses and its manifest have been written successfully.

This is **raw data capture**, not a frozen training dataset or a DVC version yet.
The collector checks response JSON and discovery structure. Full station schema,
freshness, duplicate-record, and quality checks belong to the validation stage.
The manifest's original timestamps let us add those checks without losing provenance.

## Initial modeling contract

- Observation key: station ID + source observation time. Receipt time is tracked separately.
- Proposed label: zero rentable bikes at an eligible observation around `t + 30 minutes`.
- Exact timing tolerance, operational exclusions, and cohort selection must be
  finalized after inspecting the live feed. Missing/stale labels remain unknown.
- Features may use only information received by prediction time.
- Use time-based validation with label availability respected at split boundaries.
- Compare persistence and station/time baselines before more complex algorithms.

## Verification

```bash
uv run python -m unittest discover -s tests -v
```

Tests use synthetic feeds and temporary directories, so they need no network or
AWS credentials. Live collection is a separate smoke check.

## Next guided step

Inspect two captured manifests and identify which values belong to ingestion
time versus source time. Then prepare the Mumbai S3/Lambda deployment, including
its cost estimate and resource limits, before creating cloud resources.
