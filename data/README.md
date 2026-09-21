# 📦 Data

The data files are **not stored in git**. This page explains why, what you need, and how to get it.

## What you need

| File | Size | Where it goes | How you get it |
|---|---:|---|---|
| `madrid_train.parquet` | ~142 MB | `data/` | download (see below) |
| `amsterdam_data.parquet` | ~44 MB | `data/` | download (see below) |
| `preprocessed/preprocessed_v2.pkl` | ~68 MB | `data/preprocessed/` | **generate it** by running `3Notebook_Processing.ipynb` |
| `preprocessed/preprocessed_data.pkl` | ~56 MB | `data/preprocessed/` | only needed for the original baseline notebooks. Generate it with `baseline_problem/3-Preprocessing.ipynb` |

After setup your folder should look like this:

```text
data/
├── README.md
├── madrid_train.parquet
├── amsterdam_data.parquet
└── preprocessed/
    └── preprocessed_v2.pkl      ← created by Notebook 3
```

## Option A: the download script (recommended)

```bash
python scripts/fetch_data.py
```

It downloads both parquet files from this repo's GitHub Release (tag `data-v1`), checks each file's SHA-256 fingerprint, and skips files you already have.

If the data lives somewhere else (a shared drive, a different release), point the script at it:

```bash
python scripts/fetch_data.py --base-url https://example.com/path/to/files
```

## Option B: manual

Download the two parquet files from the [Releases page](https://github.com/definitelynotironman/BuildingClassifier/releases) and drop them into `data/`. Then run `python scripts/fetch_data.py --verify-only` to check them.

## Why the data is not in git

- **GitHub blocks any file over 100 MiB** and warns above 50 MiB. `madrid_train.parquet` is about 142 MB, so a normal push fails.
- **Git remembers everything forever.** Every version of a 142 MB file you commit stays in the history and gets downloaded by every clone, even after you delete it.
- **The `.pkl` files are derived data.** They can be rebuilt from the parquet files by the notebooks, so storing them is just a second copy of the same information.
- **The datasets came from the hackathon organisers.** Check their terms before making them public anywhere.

## Checksums

Use these to confirm your copy is the same one the results were produced with.

| File | SHA-256 |
|---|---|
| `madrid_train.parquet` | `0e106a07044d867f8c68e77de8c36826c491d0f26692b8cde753d1046ce24c9c` |
| `amsterdam_data.parquet` | `32388f3a9792b01d9831ca4d0fc4bfadba132a1c23c06e372d7a82e985be0719` |

## Column cheat sheet

| Column | Meaning |
|---|---|
| `pixel_id`, `px_key`, `py_key` | Pixel identity and its position on the 30 m grid |
| `city`, `year` | Which city, which year (1984 to 2025) |
| `coverage` | Share of the pixel covered by building footprints (0.15 to 1.0) |
| `weighted_mean_year` | Area-weighted mean construction year. **The target** |
| `Blue_N` … `SWIR2_N` | The six spectral bands for snapshot N (1, 2 or 3) |
| `qa_valid_N` | True if snapshot N is clean (no cloud or shadow) |
| `doy_N`, `scene_id_N`, `dataset_N` | Day of year, Landsat scene and sensor for snapshot N |
