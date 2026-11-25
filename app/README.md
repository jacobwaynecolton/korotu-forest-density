# Canopy Cover Mapper (web tool)

Pick an area on a map, or type its corners. The tool finds recent Sentinel-2 images of that area, runs the final
model, and shows three canopy-cover maps next to the satellite image:
- **conservative**: the true cover is below it in about 10% of cells;
- **central**: the median;
- **optimistic**: the true cover is above it in about 10% of cells.

Move the pointer over any image to read the three values of a 10 m cell. Maps can be shown with a continuous colour
ramp, or as classes whose number, bounds, names and colours you choose. The tool also gives the area per class (or
the area under canopy) for each map, an overlay on the web map, and a 3-band GeoTIFF download.

## Start, stop, open

```bash
bash app/run.sh          # start (or restart) in the background on apps0, port 8765
bash app/run.sh stop     # stop
```

The server listens on `127.0.0.1` only. To open it from your computer:
- **VS Code**: in the **Ports** panel, forward port 8765.
- **Terminal on your computer**: `ssh -L 8765:localhost:8765 <your apps0 login>`.

Then open `http://localhost:8765/?token=<token>` once; the browser keeps a cookie afterwards. The token is in
`<pipeline data>/app/token.txt` (`/w/20261/tsalin/MLOps/pipeline_final/app/token.txt`). The log is
`<pipeline data>/app/server.log`.

## What happens when you run an analysis

1. **Area**: at least 500 m and at most 20 km on each side. A 10 m grid is laid out in the area's UTM zone, snapped
   to 60 m like the training data.
2. **Images**, chosen automatically in the most recent summer. The area must lie inside one Sentinel-2 tile.
   - **Window**: Sentinel-2 L2A from the training collection, 1 June to 15 September of the current summer (up to
     today) or, before June, of the last one.
   - **Seasonal choice**: up to 6 images spread over the summer, 3 before 1 August and 3 from 1 August. Each has at
     most 10% of the area flagged by the scene classification (cloud, shadow, cirrus, snow, saturated, no data). The
     cleanest come first, at least 5 days apart, and one half is filled from the other if it lacks images.
   - **Fallback**: with fewer than 4 such images, the pipeline's rule applies. That is up to 4 images, most recent
     first, under 1% flagged, with a warning. If no image of that summer passes, the two previous summers are tried,
     with a warning.
   - **Year**: a given summer can also be chosen. The number of images is not a user choice.
3. **Model**: the 5 final U-Nets (`pipeline/`) predict each image on its own (4 flips).
   - **Base maps**: each network averages the unflagged dates pixel by pixel. The 5 networks are combined as a
     mixture, and the bounds are calibrated with the final per-band constants
     (`pipeline/manifests/calibration.json`), exactly as in the evaluation.
   - **Date rule (fields)**: with the seasonal choice, two conditions are tested per pixel on the clear dates. If the
     central value of two dates differs by more than 0.35 and at least 2 dates are below 0.3, the three maps of the
     date with the lowest central value replace the base maps. Each date's maps combine the 5 networks; the
     calibration is unchanged.
   - **Why the rule**: dense green crops (hay, cereals) are read as canopy on June and July images, and much less
     once mown or harvested. The lidar labels them close to 0.
   - **Display**: the page gives the share of cells corrected this way. The settings are at the top of
     `app/engine.py`.
   - **Test**: 20 held-out test sites with the most cropland and grassland (ESA WorldCover), in the summer of their
     lidar flight. The MAE was 0.098 against 0.102 with the pipeline's image rule, and 0.026 against 0.055 on
     cropland.
   - **False clearings**: the rule changed 3.4% of the cells. 0.4% of all cells are forest in the lidar that became
     false clearings.
   - **Status**: the thresholds were chosen on these same sites, so the rule is an application heuristic, not part
     of the evaluated model.
4. **Outputs**, kept for the last 60 analyses in `<pipeline data>/app/analyses/<id>/`:
   - `canopy_cover.tif`: 3 bands, cover × 10 000, uint16, 65535 = no data;
   - one true-colour JPEG per image;
   - the maps for the web page.

Measured on apps0:

| Area | Images | Time |
|---|---|---|
| 3.3 × 3.3 km | 6 | 39 s |
| 5 × 5 km | 2 | 30 s |
| 5 × 5 km | 6 | 74 s |

The model takes about 0.4 s per km² and per image for the 5 networks, so about 2.5 s per km² with 6 images. Expect
4 to 5 minutes for 10 × 10 km, and about 17 minutes for 20 × 20 km.

## Limits

- **Speed**: apps0 gives each user at most 8 CPUs, so the 5 networks run one after another, and an area of several
  hundred km² takes minutes. Running the models on a GPU node would make this about 10 times faster.
- **One tile per area**: images from several Sentinel-2 tiles are not mosaicked.
- **Outside Ontario's forest**: the model is much less reliable there (see the NEON study); the page warns when the
  area is outside Ontario.
- **Wildfire smoke and haze**: the 1% cloud rule does not catch them.
- **Place search**: OpenStreetMap Nominatim, at most one search per second.
- **Basemaps**: OpenStreetMap, or Esri World Imagery, loaded by the browser.

## Code

- `engine.py`: area check, scene search, image reading, inference, mixture, calibration, outputs.
- `server.py`: web server and JSON API, Python standard library only.
- `static/`: the page (Leaflet for the map).
- `run.sh`: start and stop.

The API:
- `POST /api/analyses {"bbox": [W, S, E, N], "year": null}`;
- `GET /api/analyses/<id>`;
- `GET /api/analyses/<id>/<file>`;
- `GET /api/geocode?q=`.
