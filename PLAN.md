# LiDAR ground truth: plan

Owner: Jacob. Phase 1 complete (see §7 and README.md). Goal: canopy-cover labels (0–1) on a 10 m grid that teammates can pair
with Sentinel-2 L2A imagery.

## 1. Source: Ontario FRI leaf-on SPL lidar (chosen)

Why this one and not the Ontario Classified Point Cloud named in the charter:

| | FRI leaf-on SPL | Ontario Classified Point Cloud (GEO) |
|---|---|---|
| Leaf state | leaf-on (flown for forest inventory) | likely leaf-off (provincial practice); leaf-off undercounts deciduous canopy |
| Years | 2018–2025, all Sentinel-2 era | 2009–2019 per catalogue |
| Vegetation classes | yes (3/4/5/12) | no (ground/water/noise/unclassified only) |
| Ready-made CHM | yes, 0.5 m GeoTIFF | no |
| Coverage | ~452,000 km² of Ontario's managed forest zone | most of Ontario, mixed projects |

Facts verified on 2026-10-18 from the downloaded tile index and a probe tile:
- 452,122 tiles of 1 km × 1 km; tile name encodes UTM zone (15–18) and acquisition year.
- CHM: float32, 0.5 m, 2000 × 2000 px, NAD83(CSRS) / UTM zone N, zstd, ~11.6 MB per tile.
- Exact flight dates are readable from each COPC .laz header (GPS time min/max) using a
  1 KB HTTP range request, so there's no need to download point clouds (`scripts/laz_dates.py`).
- Some tiles were flown late September, when leaves may already be turning, so screen by date.

Tiles per year (all zones): 2018 21k · 2019 46k · 2020 60k · 2021 97k · 2022 75k ·
2023 77k · 2024 41k · 2025 36k. 1,080 tiles have no CHM link.

## 2. Sampling design: contiguous sites, not scattered tiles

- **Site = 5 × 5 km block of 25 contiguous tiles** from a single acquisition year and
  UTM zone.
  - Big enough for Sentinel-2 patch context (a 256 px patch is 2.56 km).
  - Makes spatial train/val/test splits clean.
- **Initial batch: 12 sites** (300 tiles, ~3.5 GB CHM), which gives ~3M labelled 10 m cells.
  This matches the charter's "hundreds of thousands to a few million" estimate.
- Spread sites across the province: west to east (UTM 15–17, and 18 if eligible) and
  south to north, to cover different forest types (Great Lakes–St. Lawrence mixedwood to
  boreal conifer).
- Spread across acquisition years.
- Eligibility:
  1. All 25 tiles exist with a CHM link, same year, same zone.
  2. Year ≥ 2019. Global Sentinel-2 L2A from late 2018 on; 2018 tiles are a fallback only.
  3. Flight dates (from LAZ headers) between **June 1 and Sept 15** (peak leaf-on).
- Split by site (spatial blocks): roughly 8 train / 2 val / 2 test, with the test sites
  geographically separated from the training sites.
- Later, if needed: add sites, or a Québec MHC (1 m) held-out test region.

## 3. Labels (next step, after download is verified)

1. Tree mask: CHM > 2 m (main; the standard "% first returns > 2 m" convention). Also
   keep a > 5 m variant (matches FAO/Canada forest definitions) for sensitivity checks.
2. Canopy cover = fraction of tree pixels per 10 m cell, computed by area-weighted
   (average) resampling of the 0.5 m mask straight onto the **Sentinel-2 grid**:
   WGS84 / UTM same zone (EPSG:326zz), 10 m pixels aligned to multiples of 10 m.
   Labels then line up with S2 pixels with no further resampling of the imagery.
3. Also output a valid-fraction band; drop cells < 100% covered (site edges, nodata).
4. NAD83(CSRS) vs WGS84 differ by ~1–2 m in Ontario. That's below 0.2 of an S2 pixel and
   below S2's own geolocation error, but handle it explicitly in the reprojection and
   document it.
5. Class cut points (dense/moderate/sparse/non-forest) are not fixed here. Store
   continuous cover and decide thresholds from EDA plus Korotu input.

## 4. Handoff to the Sentinel-2 teammates (interface)

For each site, `data/manifest/sites.geojson` gives:
- `site_id`, `split`, `utm_zone`, `epsg_s2`, bounds (in S2 CRS and lon/lat)
- `lidar_year`, `flight_start`, `flight_end` (UTC): match S2 scenes as close as possible,
  same leaf-on season, cloud-filtered
- tile list

Labels: `data/labels/<site_id>_cover10m.tif` (band 1 cover > 2 m, band 2 cover > 5 m,
band 3 valid fraction).

## 5. Risks / checks

- **Change between flight and image:** match S2 dates to the flight dates; flag sites with
  recent harvest or fire (Ontario has big fire years).
- **Leaf state:** late-season flights are screened out (§2).
- **Non-forest coverage:** the managed zone is mostly forest. Check that sites include
  lakes, cutblocks, roads and wetlands so the sparse and non-forest classes aren't empty.
  Add a site near the forest/agriculture edge if needed.
- **CHM artifacts** (power lines, buildings, spikes): rare in the managed zone; clip
  heights at a sane max and review a few sites visually.
- **Licence:** Open Government Licence – Ontario (attribution required). Cite it in the
  report.

## 6. Folder layout

```
data/raw/fri_tile_index/   tile index GeoPackage (downloaded)
data/raw/fri_chm/<site>/   CHM GeoTIFFs per site
data/manifest/             candidate list, chosen sites, flight dates
data/labels/               10 m cover labels on S2 grid
scripts/                   select_sites.py, download_chm.py, laz_dates.py, make_labels.py
```

## 7. Status (2026-10-18): phase 1 complete

Done:
- 17 sites: 12 spread out plus 5 chosen for sparse cover (probed via centre tiles).
  Flights all 1 Jun – 15 Sep, each site within 60 days; 10 candidate blocks rejected.
- 425/425 CHM tiles downloaded and decoded (4.3 GB). Three nodata conventions and a 2025
  batch in WGS84 UTM at 0.499 m are handled.
- Labels on the S2 grid for all sites (`data/labels/`), verified against an independent
  recomputation (680 cells, exact).
- One cloud-free S2 L2A preview scene per site (median 2 days from flight). All windows
  are whole-pixel aligned.
- ESA WorldCover 2021 on the label grid: water is 9.3% of cells; first third-party
  comparison done.
- Splits assigned (sites.csv `split`): test = Z15_E0415_N5390_2020, Z18_E0305_N5105_2022,
  Z16_E0350_N5600_2023; val = Z15_E0690_N5475_2020, Z17_E0400_N5110_2021,
  Z17_E0355_N5560_2019; the other 11 train. Land class mix is similar across splits.
- Report: `report/index.html`; docs: `README.md`.
- Input/label pairs (`data/pairs/`): 12-band S2 L2A reflectance per site plus a 4.13M-row
  pixel table. Found and fixed a double-applied BOA offset on baseline ≥ 04.00 scenes.
  A smoke-test per-pixel baseline (sklearn HGB, bands + NDVI, 300k train cells) gets
  val/test MAE 0.161 / 0.123, r 0.84 / 0.87, 4-class accuracy 0.67 / 0.71.

Next (team decisions first):
1. Update charter Section 3 to FRI leaf-on lidar.
2. Agree class cut points, height threshold and water handling with Korotu.
3. S2 teammates: single-scene stacks exist. Possible upgrades: multi-date composites or
   time series, bilinear rather than nearest upsampling, Sentinel-1.
4. Optional: a Québec MHC site as an out-of-province test; more sparse sites from
   `candidate_blocks.csv` if the middle classes stay weak.

## 8. Model phase (2026-10-18)

- Scaled to 49 sites (43 train) with up to 4 S2 dates each (161 training site-dates).
- U-Net ResNet-34, 3 seeds, ~5 min each. Test MAE 0.104 vs Meta CHM v2 0.148 and SCANFI v2
  0.200; wins at all 6 held-out sites. Full tables in README and the report.
- NTEMS 2022 added (test MAE 0.203 vs model 0.104). Next: Québec out-of-province test; improve the
  sparse class (more sparse sites, class-weighted loss); per-pixel uncertainty for the
  confidence map (the ensemble spread is a ready-made starting signal).

## 9. Verification (2026-10-21)

Independent checks in `verify/` (see `verify/README.md`), results in `verify/results/`,
shown in Part 3 of the report. They found one real data bug: two Sentinel-2 scenes at a
training site whose catalogue flag understated that the BOA offset was already applied,
so it was applied twice. Fixed in `make_pairs.boa_offset_applied` (pixel-based check),
site rebuilt, all seeds retrained; pre-fix results kept as `*_before_offset_fix`. Also
fixed: two dead literature links, one overstated literature sentence, and the Sentinel-2
season wording (scenes are within ±45 days of the flight, 1 May – 15 Oct; only the lidar
flights are 1 Jun – 15 Sep).
