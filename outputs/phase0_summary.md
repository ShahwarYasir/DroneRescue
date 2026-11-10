# Phase 0 findings

Dataset: AIDERv2 (16,723 image files found). Published counts check: matches the published table.

Duplicate settings: policy `pairwise`, max hash distance 5, max thumbnail difference 12.0.

## Data quality

| Check | Result |
|---|---|
| Corrupt or unreadable files | 0 |
| Images with a side shorter than 32px | 0 |
| Duplicate groups (2+ near-identical images) | 795 |
| Train images that also appear in Test (leakage) | 854 |
| Train images that also appear in Val | 803 |
| Val images that also appear in Test | 105 |
| **Test images with a look-alike in Train or Val** | **355 of 1654** (21.5%) |
| Extra copies inside Test (kept) | 100 |
| Copies inside Train/Val removed | 133 |
| Images with conflicting class labels | 2 in 1 groups (2 dropped from Train/Val) |
| Shorter side below 224px | 0.0% (median 224px) |

## Images per split and class

Before cleaning

| split | Earthquake | Flood | Normal | Wildfire | total |
|---|---|---|---|---|---|
| Train | 1927 | 4063 | 3900 | 3509 | 13399 |
| Val | 239 | 505 | 487 | 439 | 1670 |
| Test | 239 | 502 | 477 | 436 | 1654 |

After cleaning (what the models will use)

| split | Earthquake | Flood | Normal | Wildfire | total |
|---|---|---|---|---|---|
| Train | 1886 | 3970 | 3704 | 2359 | 11919 |
| Val | 237 | 502 | 477 | 347 | 1563 |
| Test | 239 | 502 | 477 | 436 | 1654 |

Test images flagged `test_leak_exposed` stay in Test; Phase 3 reports accuracy on all of Test and on the unexposed part.

Dropped files are listed in `dropped_files.csv`; duplicate groups in `duplicates_report.csv`; label conflicts in `label_conflicts.csv`.
