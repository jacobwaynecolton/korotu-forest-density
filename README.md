# Korotu forest density: LiDAR ground truth and a Sentinel-2 canopy-cover model

CSC2701 Korotu project. Canopy-cover training labels: for every 10 m Sentinel-2 pixel in
49 Ontario sites (1,219 km²), the fraction covered by tree canopy, measured from leaf-on
airborne lidar. Plus a U-Net that predicts that cover from Sentinel-2 alone, benchmarked
against Meta/WRI, SCANFI and NTEMS.

- Visual report: `report/index.html` (Phase 1 data, Phase 2 model, Part 3 verification).
- Reasoning and status: `PLAN.md`.
- **Verification:** `verify/README.md`. `.venv/bin/python -I verify/run_all.py` re-derives
  the main claims independently and prints pass or fail; `pytest tests` runs the unit tests.

Data files are not in the repo (about 35 GB, all open data). The scripts download and rebuild
them; small manifests, scores and figures are included.

## Model vs the current maps (test sites, scored once)

| 10 m, 3 test sites | MAE | RMSE | r | R² | 4-class acc. |
|---|---|---|---|---|---|
| **U-Net, 3-seed ensemble** (`models/unet_r34*`) | **0.106** | **0.167** | **0.91** | **0.82** | **74%** |
| Meta/WRI Global Canopy Height v2 (2026), cover > 2 m | 0.148 | 0.256 | 0.83 | 0.57 | 70% |
| Meta, isotonic-recalibrated on train sites | 0.150 | 0.199 | 0.87 | 0.74 | 53% |
| SCANFI v2 crown closure (2026) | 0.200 | 0.273 | 0.80 | 0.52 | 56% |
| SCANFI, recalibrated | 0.161 | 0.238 | 0.80 | 0.63 | 70% |
| NTEMS 2022 canopy cover (Matasci et al. method) | 0.203 | 0.323 | 0.68 | 0.32 | 62% |
| NTEMS, recalibrated | 0.201 | 0.280 | 0.71 | 0.49 | 46% |

- Lowest error at 5 of 6 held-out sites (NTEMS wins at Z17_E0355_N5560, val). At 30 m:
  MAE 0.081 vs 0.120 (Meta).
- Where Meta's imagery is within 2 years of the lidar (47% of held-out cells): 0.101 vs 0.119.
- Weakest class: sparse (10–40%), 42% recall, mostly confused with moderate.
- Simple baselines on the same test cells: training mean 0.34, per-pixel boosted trees on
  the 12 bands 0.15 (see `verify/check_baselines.py`).
- Caveats: 6 held-out sites, all Ontario managed forest; the model learns our label
  definition from the same lidar source (home advantage).
- Numbers are after the offset fix found in verification (`verify/README.md`). Before the
  fix: test MAE 0.104, val 0.103.
- Training: U-Net, ResNet-34 ImageNet encoder adapted to 12 bands, L1 loss, 128 px crops,
  8,000 steps, ~5 min per seed on an RTX 5070 Ti laptop GPU. Prediction averages all
  dates and 4 flips. Selection on val only.

## Results at a glance (data)

| | |
|---|---|
| Sites | 49 blocks of 5 × 5 km (12 spread, 5 chosen for sparse cover, 32 added for training ≥ 0.5° from val/test) |
| Labelled area | 1,219 km², 11.9M cells of 10 m (10.7M land, 10.3% water) |
| Lidar | Ontario FRI leaf-on SPL, flown 2019–2025, all 1 Jun – 15 Sep |
| Sentinel-2 | up to 4 cloud-free L2A dates per site (181 in all), within ±45 days of the flight, 1 May – 15 Oct |
| Class mix (land, placeholder cuts 10/40/70%) | 13% non-forest · 9% sparse · 17% moderate · 61% dense |
| Split (by whole site) | 43 train / 3 val / 3 test |
| Sentinel-2 match | every site has a < 1% cloud scene, median 2 days from the flight |
| Pairs | 4.13M (S2 L2A 12-band, cover) cells; quick per-pixel baseline on held-out sites: MAE 0.16 / 0.12, r 0.84 / 0.87 (val / test), vs 0.36 / 0.34 predicting the mean |
| Checks | 1,250/1,250 tiles decode; 3,600 random cells recomputed by an independent method, exact match; full list in `verify/` |

## Data products

`data/labels/<site>_cover10m.tif`: GeoTIFF, float32, NaN = no label.
- CRS: WGS84 / UTM zone of the site (`EPSG:326zz`, same as Sentinel-2 L2A). 10 m
  pixels, origin snapped to 60 m, so S2 10/20/60 m bands align without resampling.
- Band 1 `cover_gt2m`: fraction of the cell with canopy > 2 m (main label).
- Band 2 `cover_gt5m`: fraction with canopy > 5 m (forest-definition height).
- Band 3 `valid_frac`: fraction of the cell with lidar. Cover is only set where this is 1.

`data/pairs/`: **model-ready input/label pairs**
- `<site>_s2.tif`: Sentinel-2 L2A surface reflectance on the label grid, float32, 12 bands
  (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12). The 20/60 m bands are
  upsampled to 10 m by nearest neighbour. Scene = the cloud-free one closest to the
  lidar flight (see `pairs_manifest.csv`).
- `<site>_scl.tif`: Sentinel-2 scene classification (cloud mask).
- `pixels.parquet`: one row per 10 m cell (4.13M rows, 4.13M usable) with site, split,
  row/col, x/y, the 12 bands, SCL, `cover_gt2m`, `cover_gt5m`, WorldCover class and a
  `usable` flag (label present, bands present, not cloud/shadow/snow). Load with
  `pd.read_parquet`. For a patch model, use the GeoTIFFs and crop windows.
- Reflectance is harmonized across processing baselines. Many Element84 COGs already have
  the −0.1 BOA offset removed for baseline ≥ 04.00. Their `earthsearch:boa_offset_applied`
  flag is not always right, so `make_pairs.boa_offset_applied` also checks the pixels.

`data/manifest/`:
- `sites.csv` / `sites.geojson`: site id, split, UTM zone, `epsg_s2`, grid bounds,
  lidar flight start/end (UTC), selection reason, tile list. **This is the handoff for
  the Sentinel-2 side.**
- `s2_preview_scenes.csv`: one cloud-free S2 L2A scene per site (Element84 earth-search
  ids), the gap to the flight, and an alignment flag.
- `label_stats.csv`, `site_cover_summary.csv`, `eda_summary.json`: statistics.
- `candidate_blocks.csv`: all 14k eligible 5 × 5 km blocks, for adding sites later.

`data/context/`: S2 true-colour preview and ESA WorldCover 2021 per site, on the label
grid (EDA only, not model inputs).

## Reproduce

```bash
uv venv .venv && uv pip install --python .venv/bin/python \
    geopandas rasterio pyogrio shapely numpy pandas requests matplotlib pystac-client pyarrow scikit-learn
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python --no-deps segmentation-models-pytorch timm && uv pip install --python .venv/bin/python huggingface_hub safetensors
# tile index (334 MB) → data/raw/fri_tile_index/FRI_Leaf_On_Tile_Index.gpkg
curl -L -o data/raw/fri_tile_index/FRI_Leaf_On_Tile_Index.gpkg \
  "https://download.fri.mnrf.gov.on.ca/api/api/Download/tile-index/FRI_Leaf_On_Tile_Index_GeoPackage/FRI_Leaf_On_Tile_Index_GeoPackage.gpkg"
.venv/bin/python -I scripts/select_sites.py 12      # spread-out sites + flight-date check
.venv/bin/python -I scripts/add_sparse_sites.py 5 45 # sites chosen for 10–70% cover
.venv/bin/python -I scripts/download_chm.py         # 425 CHM tiles, ~4.3 GB
.venv/bin/python -I scripts/make_labels.py          # labels on the S2 grid
.venv/bin/python -I scripts/context_layers.py       # S2 preview + WorldCover
.venv/bin/python -I scripts/eda.py                  # splits, stats, figures
.venv/bin/python -I scripts/make_pairs.py           # S2 L2A stacks + pixels.parquet
.venv/bin/python -I scripts/check_pairs.py          # harmonization + baseline smoke test
.venv/bin/python -I scripts/expand_sites.py 33      # more training sites (then rerun download/labels/context)
.venv/bin/python -I scripts/make_stacks.py 4        # up to 4 cloud-free S2 dates per site
# NTEMS 2022 canopy cover (10 GB zip): https://opendata.nfis.org/downloads/forest_change/CA_canopy_cover_2022.zip
#   unzip into data/raw/ntems/
.venv/bin/python -I scripts/benchmarks.py all meta_chm_v2 scanfi_v2 ntems_2022
.venv/bin/python -I scripts/calibrate_benchmarks.py meta_chm_v2 scanfi_v2 ntems_2022
.venv/bin/python -I scripts/train_unet.py --name unet_r34 --seed 0 --steps 8000 --eval_every 500
for s in 1 2; do .venv/bin/python -I scripts/train_unet.py --name unet_r34_s$s --seed $s --steps 8000 --eval_every 1000; done
.venv/bin/python -I scripts/ensemble.py unet_r34_ens3 unet_r34 unet_r34_s1 unet_r34_s2
S="model:unet_r34_ens3 meta_chm_v2 meta_chm_v2_cal scanfi_v2 scanfi_v2_cal ntems_2022 ntems_2022_cal"
.venv/bin/python -I scripts/evaluate.py val $S && .venv/bin/python -I scripts/evaluate.py test $S
.venv/bin/python -I scripts/final_analysis.py unet_r34_ens3 ntems_2022
.venv/bin/python -I scripts/uncertainty_preview.py unet_r34_ens3 unet_r34 unet_r34_s1 unet_r34_s2 --site Z15_E0415_N5390_2020
uv pip install --python .venv/bin/python pytest && .venv/bin/python -I verify/run_all.py
.venv/bin/python -I report/build_report.py          # report/index.html
```

## Things to know

- **Source switch.** The charter names the Ontario *Classified Point Cloud*. That data is
  2009–2019, probably leaf-off, and has no vegetation class. FRI leaf-on lidar fixes all
  three. Update the charter's Section 3.
- **Tile formats vary:** three nodata values (−9999, −32767, FLT_MIN), and 2025 tiles
  are in WGS84 UTM at 0.499 m, 2004 px. `make_labels.py` handles all of them.
- **Flight dates** come from COPC .laz headers via a 1 KB range request
  (`scripts/laz_dates.py`). FRI files leave the LAS "adjusted GPS time" bit unset but
  store adjusted time, which the script detects.
- **Datum:** NAD83(CSRS) → WGS84 uses PROJ's Helmert (~1 m accuracy); the shift here is
  ~0.3 m E, ~1.0 m N, i.e. 0.1 S2 pixel.
- **Open questions for Korotu:** density cut points; 2 m vs 5 m threshold (do shrubs
  count?); how to treat water; whether production imagery is Sentinel-2 or sub-metre.

Note: the first seed was trained as `unet_r34` (seed 0); the ensemble is
`unet_r34 + unet_r34_s1 + unet_r34_s2`. One expansion site (Z16_E0275_N5330_2021) had no
cloud-free scene and was dropped.

## Licences

Ontario FRI lidar: Open Government Licence – Ontario. ESA WorldCover: CC BY 4.0.
Sentinel-2: Copernicus free and open data policy. Natural Earth: public domain. Meta/WRI CHM v2: CC BY 4.0. SCANFI and NTEMS: Open Government Licence – Canada.
