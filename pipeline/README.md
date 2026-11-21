# Final pipeline: probabilistic canopy cover from Sentinel-2, trained on about 250 lidar sites

For every 10 m pixel, the final models give three canopy-cover maps (fraction of the cell under canopy above 2 m):
- a **conservative** map: the 10th percentile, calibrated;
- a **central** map: the median;
- an **optimistic** map: the 90th percentile, calibrated.

The labels come from Ontario's leaf-on airborne lidar (FRI CHM, 0.5 m). The inputs are Sentinel-2 L2A images taken
during the leaf-on season.

This folder is self-contained: it does not import the first version's `scripts/`. The first version and the
studies that led to these choices stay in the repository for the record:
- `modeles-probabilistes/`: model choice and calibration;
- `independant-verification/`: GEDI, new Ontario sites, NEON, season window, BOA offset.

## What changed from the first version, and why

| Change | Why (evidence) |
|---|---|
| **One site rule**: lidar rules plus 15 km between centres, drawn at random, instead of three selection scripts with different spacing rules (0.3° or 0.5°) | Simpler. Opens a bank of 14,015 eligible blocks; about 1,150 of them are at least 15 km apart. |
| **More partial cover**: draw stratified on an NTEMS estimate, 50% from the third of blocks with the most partial cover (10–70%), 30% / 20% from the others | On our 64 sites, the NTEMS share ranks sites like the lidar label (rank correlation 0.75; 17 of 22 top-tercile sites right). The first version had to add sparse sites by hand. |
| **The 49 project sites and the 15 verification sites are in the bank** | Their data are reused (CHM cache). The test set is drawn from the whole bank. |
| **Test set and folds by spatial groups** (100 km cells) with a 25 km buffer, balanced on lidar partial cover | With 15 km spacing, the spacing rule no longer separates training from evaluation; the groups do. Sparse sites end up in both training and test. |
| **No fixed validation sites, no checkpoint choice**: fixed step count, chosen once on fold 0 among 8k, 16k, 32k | Three validation sites were too noisy to choose a checkpoint (`modeles-probabilistes/`). The first version's 8,000 steps were tuned for 43 sites. |
| **Sentinel-2 window 1 June – 15 September** (was 1 May – 15 October) | Scenes from May or after mid-September have more error in 11 of 12 cases, and underestimate cover (`independant-verification/fenetre-saison/`). |
| **Scene list frozen at selection**, images downloaded from that list only, with safe resume | The first `make_stacks.py` searched the catalogue again and treated listed sites as done, so it could not rebuild the dataset. |
| **SCL saved with each image** (13th band); flagged pixels are left out of the loss and of the date average | Rigour. The 1% cloud threshold per scene is kept. |
| **BOA offset**: one rule (reflectance = DN × 0.0001) and an alarm, instead of a three-branch detection | In the 254 scenes we used, the offset was never added; the pixel test decided only 2 scenes (`independant-verification/audit-offset-boa/`). |
| **Explicit no-data rule for the CHM** (-9999, -32767, FLT_MIN even when undeclared), with missing pixels counted | A guard for new tiles: in the 1,225 tiles we have, no pixel uses these values. |
| **Label grid from the block's nominal extent**, not from the downloaded files | Known before any download, so scenes can be chosen at selection. Identical to the first version's grid for 46 of 49 sites; the other 3 differ by one 60 m row or column. |
| **Sites without a usable label or image are dropped automatically**, and every manifest comes from `sites.csv` | In the first version, a dropped site stayed in several files and in the reported area. |
| **Downloaded CHM tiles deleted once the label is written** | About 265 MB per site; the disk only holds the sites being processed. |
| **Model**: ResNet-34 ImageNet U-Net, q10/q50/q90, pinball loss, 5 seeds combined as a mixture, per-band conformal calibration | The best configuration in site-level cross-validation (`modeles-probabilistes/`). |
| **Data kept in CPU memory as uint16**; only batches go to the GPU | The first version held every site on the GPU, which does not scale to 250 sites. |
| **Prediction for any area** (`predict_area.py`) | The first version could only predict its own sites. |

What the final pipeline leaves out of the first version:
- the single-image path (`make_pairs.py`, `pixels.parquet`, `check_pairs.py`, `check_sites.py`);
- the checks that always agree (independent label recompute, metric recompute, reading the training code by
  regular expression), which are replaced by unit tests;
- `train_unet.py`, `ensemble.py` and `uncertainty_preview.py`.

## Steps

| Step | Script | Where | What | Output |
|---|---|---|---|---|
| 1 | `01_bank.py` | CPU | eligible blocks, NTEMS partial-cover score, strata | `manifests/bank.csv` |
| 2 | `02_select_sites.py` | CPU, network | existing sites, then stratified random draw; flight dates and Sentinel-2 scenes checked for each candidate | `manifests/sites.csv`, `tiles.csv`, `s2_scenes.csv`, `selection_log.csv` |
| 3 | `03_labels.py` | CPU array, network | CHM tiles → 10 m labels, tiles deleted after use | `data/labels/`, `data/status/labels/` |
| 4 | `04_images.py` | CPU array, network | frozen scenes → 13-band stacks | `data/stacks/`, `data/status/images/` |
| 5 | `05_splits.py` | CPU | usable sites, spatial groups, test set, 5 folds | `manifests/site_status.csv`, `splits.csv` |
| 6 | `06_train.py --mode steps` | GPU array | fold 0, one seed, each step-count candidate | `data/predictions/steps/` |
| 7 | `07_choose_steps.py` | CPU | smallest step count within 1% of the best quantile score | `manifests/steps.json` |
| 8 | `06_train.py --mode cv` and `--mode final` | GPU arrays | 5 folds × 5 seeds; 5 final models on all training sites | `data/predictions/cv/`, `final/`, `data/models/` |
| 9 | `08_evaluate.py` | CPU | mixtures, calibration (per fold, then final), cross-validation and test metrics | `manifests/calibration.json`, `results/` |
| 10 | `09_maps.py` | CPU | calibrated 3-band maps of every site, from models that never saw it | `data/maps/` |
| — | `predict_area.py` | GPU or CPU | maps for any area and year, with the final models | one GeoTIFF |

On Slurm, from comps1:

```bash
bash pipeline/run.sh --dry-run     # show the jobs
bash pipeline/run.sh               # submit steps 1 to 10, each waiting for the ones it needs
bash pipeline/run.sh --from 5      # resume from a step; finished work is skipped
```

Every rule and number lives in `config.toml`. For a test on a few sites, copy it, change the paths and sizes,
and set `PIPELINE_CONFIG` to the copy. Unit tests: `.venv/bin/python -m pytest -q pipeline/tests`.

`manifests/` (site list, scene list, splits, step count, calibration constants) and `results/` are small and go in
git once the run is done: they freeze the dataset and the model. Large files go to `paths.data`.

## Disk space and time (estimate for 250 sites)

Sizes are measured on the current data:
- CHM: 265 MB per site;
- label: 1.4 MB per site;
- Sentinel-2 image: 2.6 MB per scene, uint16 with SCL (3.8 MB in the first version's float32);
- prediction: 1 MB per site and run;
- model: 47 MB.

| Item | Count | Size |
|---|---|---|
| CHM tiles of the 186 new sites (downloaded, deleted after each label) | 49 GB transferred | **≈ 2–3 GB at the peak** (8 tasks at once) |
| CHM of the 64 existing sites | already on disk, read only | 0 |
| Labels | 250 sites | 0.35 GB |
| Sentinel-2 stacks | ≈ 3.3 scenes per site, ≈ 825 scenes | ≈ 2.1 GB |
| Predictions (step candidates, cross-validation, final) | ≈ 1,400 files | ≈ 1.4 GB |
| Calibrated maps | 250 sites | ≈ 0.4 GB |
| Final models | 5 | 0.25 GB |
| Tile index and NTEMS | already on disk | 0 |
| **Total new** | | **≈ 5 GB kept, ≈ 8 GB at the peak** |

With `keep_chm = true`, add 49 GB.

**Memory**: a training job holds every training site in RAM, about 5 GB as uint16. Jobs ask for 32 GB.

**Time**, if nodes are free. Data steps are measured on a 4-site test run; training is extrapolated.

| Steps | Time |
|---|---|
| Bank | 10 min |
| Selection | about 1 h: 10 s per accepted candidate, 2 to 3 s per rejected one |
| Labels | under 30 min: about 15 s per new site (25 tiles), 8 tasks at once |
| Images | under 15 min: about 2 s per scene |
| Training | about 30 to 60 min per run at 32,000 steps; 3 + 25 + 5 runs, 6 at a time: about 5 to 8 h |
| Evaluation and maps | under 1 h |
