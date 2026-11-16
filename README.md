---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

<!-- dataset-factory:links -->
**Dataset Factory · v-75d63061fc7d7395**

[Dataset website](https://earino.github.io/factory-usgs-streamgage-record-survival/versions/v-75d63061fc7d7395/) · [GitHub release](https://github.com/earino/factory-usgs-streamgage-record-survival/releases/tag/v-75d63061fc7d7395) · [Hugging Face](https://huggingface.co/datasets/earino/factory-usgs-streamgage-record-survival/tree/v-75d63061fc7d7395) · [Catalogue](https://earino.github.io/dataset-factory-site/)

Artifact: be72de55eb39dc02a4465ad1a926ab93deb921b7c5431f21d9bbb166ded7998a
<!-- /dataset-factory:links -->

# Five-year continuation of newly established USGS daily streamflow records

When a new USGS daily mean streamflow record opens at a monitoring location, will that record still be active five years later?

Derived from U.S. Geological Survey (USGS), U.S. Department of the Interior. This is not its official product.

Prediction time: The series begin date (first day of the daily mean discharge record), using only site metadata and facts about the record and the surrounding network at the moment it opens; no later observation of that same series is used as an input.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.7181944555831795,
    "brier": 0.17094465093118094
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.18819661358878198
  },
  "calendar_identity": {
    "auc": 0.5645200096238672,
    "brier": 0.24113307202371798
  }
}
```

## Limitations

- Current-vintage retrospective design: drainage area, altitude, site type and the primary/approval flag are read from today's record, not from an archived 2001-2019 snapshot. They are plausible establishment-time attributes but this does not prove historical operational availability.
- A daily mean discharge series can end because publication changed (gage continues with stage only, converts to instantaneous-only, or is transferred to another agency), not only because the physical gage closed. The label is record continuation, not infrastructure survival.
- Series with internal gaps count as continued if the last observation passes the fifth anniversary; no data-completeness threshold is applied, which the next task may add as a documented secondary variant.
- USGS `begin` dates for older or migrated sites can reflect data-system history rather than the first day of physical operation; restricting to 2001-2019 reduces but does not eliminate this.
- Cooperating-agency and non-USGS records are excluded by the agency_code = 'USGS' filter, so the population is USGS-operated records only; this also keeps the statutory public-domain basis clean.
- 17 U.S.C. 105 is a U.S.-law exclusion and the legislative history states it has no effect on protection abroad; we claim U.S. public-domain status only and assert no worldwide dedication.
- Unauthenticated API rate limits apply; an api_key may be needed for bulk paging, and the service's page size and cursor behaviour should be handled defensively rather than assumed.
- Labels freeze at the 2026-11-15 snapshot; all outcome windows close by 2024-12-31, so late data revisions should not change this cohort's labels, but a later rebuild will produce a slightly different population.
- State and district features partly encode funding and program shocks, so the task measures administrative persistence as well as hydrologic or infrastructure factors; no causal interpretation is warranted.
- Current-vintage retrospective design: drainage area, contributing area, altitude, site type, state, district and the primary flag are read from the current record, not an archived 2001-2019 snapshot, so their operational availability at establishment is unverified.
- The label is record continuation, not physical gage survival: a series can end because publication changed, the site converted to stage-only, or the record moved to another agency.
- The 365 day continuation exclusion can only see sibling series that begin inside the 2001-2019 download window, so a location whose earlier series began before 2001 and overlapped the new begin date cannot be detected from this extract.
- The nearby gage count and the regional termination rate are computed from series present in this single download, which excludes cooperating-agency records and series beginning outside the window.
- No full-row availability ledger exists, so training label timing is unverified and the task is not a forecasting claim.
- Missing site metadata such as drainage area, contributing area and altitude stay missing rather than being imputed as zero.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
