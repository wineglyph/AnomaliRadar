# Datasets

## MetroPT-3 (primary dataset)

| Field | Value |
| --- | --- |
| Name | MetroPT-3 |
| Repository | UCI Machine Learning Repository |
| Dataset ID | 791 |
| Source page | <https://archive.ics.uci.edu/dataset/791/metropt+3+dataset> |
| Download (direct) | <https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip> |
| DOI | 10.24432/C5VW3R |
| License | **CC BY 4.0** (verified on the UCI dataset page, 2026-11-16) |
| Redistribution | Allowed with attribution. **We still do not vendor it** — see below. |
| Archive size | 218,381,995 bytes (≈208 MB) |
| SHA-256 | `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a` |

### Attribution / citation

> Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). *MetroPT-3 Dataset*
> [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5VW3R>

Associated papers:

1. Davari, N., Veloso, B., Ribeiro, R.P., Pereira, P.M., Gama, J. (2021).
   *Predictive maintenance based on anomaly detection using deep learning for air
   production unit in the railway industry.* IEEE DSAA 2021.
   <https://doi.org/10.1109/DSAA53316.2021.9564181>
2. Veloso, B., Ribeiro, R.P., Pereira, P.M., Gama, J. (2022).
   *The MetroPT dataset for predictive maintenance.* Scientific Data 9, 764.
   <https://doi.org/10.1038/s41597-022-01877-3>

### Download instructions

```bash
python scripts/download_data.py          # download + extract + write manifest
python scripts/download_data.py --verify-only
```

The downloader pins the official archive SHA-256
`aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`.
`--verify-only` checks the archive against both the pinned value and the recorded
manifest. A download whose checksum does not match is rejected before the
experiment is run.

The script writes:

```
data/raw/metropt3.zip                     # archive (git-ignored)
data/raw/manifest.json                    # checksum + retrieval timestamp
data/raw/metropt3/MetroPT3(AirCompressor).csv
data/raw/metropt3/Data Description_Metro.pdf
```

`data/` is git-ignored. Nothing in this repository redistributes the dataset.

### Data schema (as actually present in the CSV)

The archive contains **17 columns**: an unnamed row counter, `timestamp`, 7 analog
and 8 digital channels. Row 0's counter is 0 and each row increments by 10, i.e.
the counter is the timestamp in **deciseconds**, not a 1 Hz sample index.

| # | Column (in file) | Type | Unit | Notes |
| --- | --- | --- | --- | --- |
| 0 | `Unnamed: 0` | int | 0.1 s | Row counter, equals `10 × row_index`. Drop it. |
| 1 | `timestamp` | str → datetime | — | `YYYY-MM-DD HH:MM:SS`, no timezone, no DST handling. |
| 2 | `TP2` | float | bar | Compressor pressure. **≈0 whenever not loaded.** Has a −0.012 baseline offset. |
| 3 | `TP3` | float | bar | Pneumatic panel pressure. |
| 4 | `H1` | float | bar | Cyclonic separator pressure drop. Small negative offset. |
| 5 | `DV_pressure` | float | bar | Air-dryer tower discharge pressure drop. Small negative offset. |
| 6 | `Reservoirs` | float | bar | Downstream reservoir pressure; tracks `TP3`. |
| 7 | `Oil_temperature` | float | °C | Range 15.4 – 89.05. |
| 8 | `Motor_current` | float | A | ~0 off, ~4 offloaded, ~6 loaded, ~8.5 start. |
| 9 | `COMP` | float 0/1 | — | Intake valve signal (active = no intake). |
| 10 | `DV_eletric` | float 0/1 | — | Outlet valve (active = under load). |
| 11 | `Towers` | float 0/1 | — | Tower 1 / tower 2 selector. |
| 12 | `MPG` | float 0/1 | — | Start-under-load trigger; agrees with `COMP` 99.57% of rows. |
| 13 | `LPS` | float 0/1 | — | Active when pressure < 7 bar. Rare (0.34%). |
| 14 | `Pressure_switch` | float 0/1 | — | Dryer tower discharge detection. |
| 15 | `Oil_level` | float 0/1 | — | Active when oil is **below** expected level. |
| 16 | `Caudal_impulses` | float 0/1 | — | Air-flow pulse output. |

> **Naming discrepancy (verified):** the UCI web page and the PDF write
> `DV_electric`, `TOWERS`, `Caudal_impulse`. The CSV uses `DV_eletric`, `Towers`,
> `Caudal_impulses`. The CSV is authoritative; `tests/test_data_schema.py` maps the
> documented spellings via `COLUMN_ALIASES`.

> **Documentation discrepancy (verified):** the PDF states "15169480 data points
> collected at 1Hz". The file has 1,516,948 rows at a **10 s median interval**.
> The PDF figure is the decisecond counter's max value; the true sampling rate is
> 0.1 Hz. Do not use the PDF's "1Hz" claim.

### Redistribution policy

Code is MIT. The dataset is CC BY 4.0 and is **never committed**. Reasons:

1. Keeps the repository small and cloneable.
2. Avoids stale copies drifting from the officially versioned artifact.
3. Makes the licence provenance explicit at the point of download.

Any figure or table derived from the data is attributed to the citation above and
is accompanied by its generating script.

## Considered alternates (not used on Day 0)

| Dataset | Licence | Why not primary |
| --- | --- | --- |
| NAB | MIT repo; per-series provenance varies | Retained as a possible secondary breadth set. |
| SKAB | GPL-3.0 | Copyleft on data; only usable via runtime download. |
| Bike Sharing (UCI 275) | CC BY 4.0 | No anomaly labels; needs synthetic injection. |
