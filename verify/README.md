# Verification

Each script re-derives one claim from the report by a route that does not reuse the
pipeline's code. Outputs land in `verify/results/*.json`, and the report reads its Part 3
numbers from those files.

```
.venv/bin/python -I verify/run_all.py           # everything, ~15 min (label recompute is the slow part)
.venv/bin/python -I verify/run_all.py --quick   # fewer label cells, ~4 min
.venv/bin/python -m pytest -q tests             # unit tests only
```

| Script | Claim | Independent route |
|---|---|---|
| `check_inputs.py` | Inputs are sane | Same grid as label; no negative reflectance; forest pixels have vegetation-like NIR; scene dates |
| `check_labels.py` | Label = share of 0.5 m CHM pixels > 2 m per 10 m cell | Projects each of the 400 sub-pixel centres into the tile CRS with pyproj and reads the CHM by index. No GDAL warp. `check_labels.py 60 1` is the negative control (1 m shift must break the match) |
| `check_alignment.py` | Labels sit on the Sentinel-2 pixels | Correlation of label vs NDVI for shifts of ±3 px; peak location plus sub-pixel parabola fit |
| `check_splits.py` | No train/test leakage | Footprint overlaps, shared tiles, distance to nearest training site; reads `train_unet.py` for where normalization, crops and checkpoint choice come from |
| `check_metrics.py` | Metric tables are right | scikit-learn on freshly read rasters vs `data/manifest/eval_*.json` |
| `check_benchmarks.py` | Benchmarks read and converted correctly | Single-pixel reads from the original NTEMS/SCANFI files; Meta cover recomputed from its original height tile |
| `check_baselines.py` | How much the U-Net adds | Constant, NDVI linear, and per-pixel gradient boosting on the same cells |
| `list_runs.py` | Every training run is disclosed | Reads every `models/*/config.json` |

## What the checks found

1. **Double-applied reflectance offset (fixed).** At training site `Z16_E0420_N5540_2022`, the
   catalogue flag `earthsearch:boa_offset_applied` was `False` for 2 of 4 scenes, but their
   pixels already had the −0.1 offset removed, so it was subtracted twice (reflectance
   ≈ −0.09). Found by `check_baselines.py`, where the NDVI fit collapsed, and confirmed by
   `check_inputs.py`. `make_pairs.boa_offset_applied` now also checks the pixels: with the
   offset still in, blue DN ≥ 1000 for real surfaces. The site was rebuilt and all three seeds
   retrained. Pre-fix results are kept as `data/manifest/*_before_offset_fix.json`.
2. **Literature links.** Two dead links were replaced (a DOI, and an archive.org copy of the
   USFS lidar metrics guide). One sentence claimed more than its source; it was rewritten.
3. **Sentinel-2 season wording.** The 1 Jun – 15 Sep window applies to lidar flights only.
   Sentinel-2 scenes are within ±45 days of the flight, 1 May – 15 Oct.
4. **Sub-pixel offset.** Median leftover label/image offset of about 2–3 m (a quarter pixel).

## Limits

The checks show the labels, inputs and scores are computed as described. They don't show
that the model generalizes outside Ontario's managed forest, and the held-out set is only
6 sites. The checks were written by the pipeline's author, so code review by someone else is
still worthwhile.
