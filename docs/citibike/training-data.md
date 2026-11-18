# Local training-data preparation

`prepare_training.py` turns cleaned station history into input/answer pairs.
It does not fit a model, deploy anything, upload data, or resume collection.

## How one example is created

1. Treat a completed collection as a possible prediction moment for each accepted station.
2. Recheck the latest report's age at that moment and whether the station allows rentals.
3. Look back to exactly 15 minutes before that moment. Use the most recent row
   whose collection had completed by that earlier cutoff, including excluded rows
   when checking whether the latest state was usable. A report downloaded after
   the cutoff cannot be a historical input just because its source timestamp is older.
4. Require that historical row to be usable and no more than 15 minutes old at
   the historical cutoff. Do not fill missing history or jump past a bad latest row.
5. Find the station report nearest prediction time +30 minutes, within +/-2.5
   minutes by default. Equal-distance ties choose the earlier report. If the
   nearest report is excluded or otherwise unusable, skip rather than choosing
   a farther valid report. Zero bikes at an operating station means target 1;
   a positive count means target 0. Missing observations never mean zero bikes.
6. Save the example only once the available collection history extends beyond
   the whole matching window. The last part of a download therefore has no labels yet.

The default 15-minute freshness rule and +/-2.5-minute future matching window
serve different purposes. `--max-age-minutes` controls input/report age;
`--label-tolerance-minutes` controls how far the matched outcome may be from +30 minutes.
The target approximates the state near +30 minutes; it does not establish the
exact state between observed reports or the probability of becoming empty at any point.

Repeated downloads of the same source report are kept as possible prediction
moments. For labeling, the report's first received copy is used. Conflicting bike
counts or rental/installation states at the same reported timestamp make that
report unusable as a target; later stale repeats alone do not erase an original
usable report.

## Important assumption about prediction time

`prediction_at = collection_completed_at` is a local replay assumption: the
collector has finished writing local files. Both source responses must have
arrived by then. This timestamp does not guarantee a future cloud API already
has access to the files or processed features. Cloud serving must record its
actual input availability time before reusing these rules.

Report age is recomputed at prediction time. This can differ slightly from the
cleaner's age at receipt. Future-dated input reports are rejected here even if
they fell within the cleaner's 60-second clock allowance.

## Selecting the initial station group

Up to 100 station IDs are chosen from the earliest accepted batch, sorted as text.
The exact IDs are saved in `dataset_manifest.json`. This is a reproducible starting
group, not a representative sample of all neighborhoods. New station arrivals and
future targets do not change the initial selection. `--station-limit` changes its size.

## Model inputs: the explicit allowlist

The future trainer must select `dataset_manifest.json["feature_columns"]`, rather
than using every column other than the target. Several extra columns contain
future information and must never be inputs.

| Input column | Meaning and origin |
| --- | --- |
| `station_id` | Station identity from the source; treat as a category when modeling |
| `num_bikes_available` | Latest accepted reported bike count |
| `num_docks_available` | Latest accepted reported dock count |
| `capacity` | Matching station-information record's capacity |
| `is_returning` | Whether the station allows returns; rentals must already be enabled |
| `report_age_seconds` | Prediction time minus the current station report time |
| `hour_of_day` | Prediction hour in `America/New_York`, including daylight saving time |
| `day_of_week` | New York weekday: Monday=0, Sunday=6 |
| `bikes_known_15min_ago` | Bike count known at the historical cutoff, not necessarily counted exactly then |
| `lag_report_age_seconds` | Historical cutoff minus the historical station report time |
| `bike_change_15min` | Current known bike count minus the count known at that cutoff |

`empty_after_30min` is the binary answer, not an input. Neither document-level
feed update timestamp is included in the model-input list.

## Extra columns: tracing and checking, not inputs

| Column | Why it is saved |
| --- | --- |
| `prediction_at` | Identifies the example's prediction moment and supports time-based splitting |
| `current_reported_at` | Shows when the current bike count was reported |
| `lag_available_at` | Proves the historical batch was available before the cutoff |
| `lag_reported_at` | Shows the historical count's actual reported time |
| `target_at` | Desired prediction time +30 minutes |
| `label_reported_at` | Actual source time of the matched future observation |
| `label_available_at` | When the matched observation's batch completed; needed to respect label availability in splits |
| `current_collection_id` | Links the input to its original cleaned collection |
| `lag_collection_id` | Links the historical input to its collection |
| `label_collection_id` | Links the answer to its collection |
| `label_num_bikes_available` | Future count explaining why the target is 0 or 1; never a feature |
| `label_time_error_seconds` | Matched report time minus desired target time; never a feature |

## Outputs

| File | Contents |
| --- | --- |
| `training_examples.parquet` | One combined table of features, target, and tracking columns |
| `skipped_examples.parquet` | Accepted candidate observations that could not become training examples, with reasons |
| `training_report.md` | Readable counts and the rules actually used |
| `training_report.json` | The same settings and counts for programs |
| `dataset_manifest.json` | Input/output file hashes, builder hash, selected stations, feature list, target, and settings |

The builder reads accepted and excluded rows, but only accepted rows become
prediction candidates. Excluded rows are retained to prevent substituting old
good information for a known bad/closed state during matching. A run with no usable
examples still saves an empty typed table and a report marked `insufficient_usable_examples`.

Outputs are built in a temporary folder and published together when complete.
Input hashes are rechecked before publishing to detect changes during the run.
The manifest records the files used; it is not a DVC version or a cloud backup.

Next, review the report and then define chronological training/validation/test
boundaries. Exclude examples whose future matching window or label availability
crosses the next split boundary. Do not randomly mix earlier and later examples.
