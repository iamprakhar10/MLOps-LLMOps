# Two stations: complete downloaded record examples

These examples were extracted from an existing local collection; no new data was downloaded.
Original collection: `20261109T104610176600Z_a2df445e71a9404e94c10abb7081eec9`.

- The two station JSON examples preserve every field present in these two records and all original top-level fields.
- Only the `data.stations` arrays were shortened to two stations. The original feeds contained 2,520 station records each.
- Discovery is copied in full. It lists other URLs; our collector downloads only discovery, station status, and station information.
- Original raw files and manifest were not changed. These excerpts do not match the original manifest sizes/hashes and are not a replacement raw collection.
- These are fields in this saved GBFS 1.1 collection. Future feed records may contain different or additional fields.

## The three times

The `_at` names below belong to the cleaned table. The downloaded documents use `last_updated` and `last_reported`.

| Cleaned column | Original field | Station A | Station B |
| --- | --- | --- | --- |
| `information_feed_updated_at` | `station_information.last_updated` | 2026-11-09 16:16:10 IST | Same document, same value |
| `status_feed_updated_at` | `station_status.last_updated` | 2026-11-09 16:15:11 IST | Same document, same value |
| `last_reported_at` | Each station status record's `last_reported` | 2026-11-09 16:11:17 IST | 2026-11-09 16:12:38 IST |

The first two times belong to whole documents. Each document contains both stations. The third time belongs to an individual station record.
A feed update does not mean every station reported at that time, and does not necessarily mean station details changed.

## Original document structure

Every example preserves the top-level `last_updated`, `ttl`, `version`, and `data` fields.
`ttl` is the feed time-to-live value in seconds; it is not the 15-minute age threshold chosen by our cleaner.
Discovery stores `data.<language>.feeds`, with each entry containing `name` and `url`.
The station documents store `data.stations`, a list of station records.

## Station information: every record field

| Downloaded field | Station A | Station B |
| --- | --- | --- |
| `station_id` | `"2235290248874043146"` | `"66de2b2f-0aca-11e7-82f6-3863bb44ef7c"` |
| `name` | `"Linden Blvd & E 91 St"` | `"3 St & Hoyt St"` |
| `short_name` | `"3437.03"` | `"4110.10"` |
| `lon` | `-73.9208` | `-73.99364123` |
| `lat` | `40.6546` | `40.6777287` |
| `region_id` | `"71"` | `"71"` |
| `capacity` | `12` | `53` |
| `rental_uris` | `{"android": "https://bkn.lft.to/lastmile_qr_scan", "ios": "https://bkn.lft.to/lastmile_qr_scan"}` | `{"android": "https://bkn.lft.to/lastmile_qr_scan", "ios": "https://bkn.lft.to/lastmile_qr_scan"}` |
| `rental_methods` | `["KEY", "CREDITCARD"]` | `["KEY", "CREDITCARD"]` |
| `has_kiosk` | `false` | `true` |
| `station_type` | `"classic"` | `"classic"` |
| `external_id` | `"2235290248874043146"` | `"66de2b2f-0aca-11e7-82f6-3863bb44ef7c"` |
| `electric_bike_surcharge_waiver` | `false` | `false` |
| `eightd_station_services` | `[]` | `[]` |
| `eightd_has_key_dispenser` | `false` | `false` |

## Station status: every record field

| Downloaded field | Station A | Station B |
| --- | --- | --- |
| `station_id` | `"2235290248874043146"` | `"66de2b2f-0aca-11e7-82f6-3863bb44ef7c"` |
| `num_bikes_available` | `10` | `45` |
| `num_bikes_disabled` | `2` | `6` |
| `num_docks_available` | `0` | `2` |
| `num_docks_disabled` | `0` | `0` |
| `is_installed` | `1` | `1` |
| `is_renting` | `1` | `1` |
| `is_returning` | `1` | `1` |
| `last_reported` | `1794220877` | `1794220958` |
| `legacy_id` | `"2235290248874043146"` | `"3381"` |
| `num_ebikes_available` | `2` | `21` |
| `num_scooters_available` | `0` | `0` |
| `num_scooters_unavailable` | `0` | `0` |
| `eightd_has_available_keys` | `false` | `false` |

## Download metadata created by our collector

`manifest.json` is created by our code, not downloaded from Citi Bike. It records collection start/end, source and resolved URLs, request and receipt times, source update time, source TTL, GBFS version, file names, compressed/uncompressed sizes, hashes, and station counts for the station feeds.
It remains separate from these example source documents. Neither the downloaded rows nor the original manifest contains our future training target.

## Files

- [Complete station-information example](station_information.example.json)
- [Complete station-status example](station_status.example.json)
- [Complete discovery document](discovery.example.json)
