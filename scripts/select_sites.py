"""Pick 5x5 km lidar sites from the FRI leaf-on tile index.

Site rules (see PLAN.md §2): 25 contiguous 1 km tiles, one UTM zone, one acquisition
year >= MIN_YEAR, every tile has a CHM, flight dates within the leaf-on window.
Sites are spread across the province with farthest-point sampling on block centres.

Usage: python -I scripts/select_sites.py [n_sites]
"""
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from laz_dates import copc_time_range  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/raw/fri_tile_index/FRI_Leaf_On_Tile_Index.gpkg"
OUT = ROOT / "data/manifest"
BLOCK_KM = 5
MIN_YEAR = 2019
LEAF_ON = ((6, 1), (9, 15))  # inclusive month/day window
MAX_SPAN_DAYS = 60  # all flights of a site within one season
SEED = 2701
# NAD83(CSRS) / UTM zone N (the FRI CHM CRS); codes are not sequential
CSRS_UTM_EPSG = {15: 3159, 16: 3160, 17: 2958, 18: 2959}


def parse_tiles() -> pd.DataFrame:
    df = pyogrio.read_dataframe(INDEX, layer="FRI_Tile_Index", read_geometry=False)
    t = df.Tilename  # e.g. 1kmZ17 3420 5485 0 2021 L
    df["zone"] = t.str[4:6].astype(int)
    df["e_km"] = t.str[6:10].astype(int) // 10   # stored in 100 m units
    df["n_km"] = t.str[10:14].astype(int)        # lower-left northing, km
    df["year"] = t.str[15:19].astype(int)
    return df


def candidate_blocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.Download_CHM.notna() & (df.year >= MIN_YEAR)].copy()
    df["be"] = df.e_km // BLOCK_KM * BLOCK_KM
    df["bn"] = df.n_km // BLOCK_KM * BLOCK_KM
    g = df.groupby(["zone", "year", "be", "bn"])
    blocks = g.size().rename("n").reset_index()
    blocks = blocks[blocks.n == BLOCK_KM**2].reset_index(drop=True)
    # block centre in lon/lat for spreading sites out
    lon = np.empty(len(blocks)); lat = np.empty(len(blocks))
    for z, idx in blocks.groupby("zone").groups.items():
        tr = Transformer.from_crs(f"EPSG:{CSRS_UTM_EPSG[z]}", "EPSG:4326", always_xy=True)
        b = blocks.loc[idx]
        x = (b.be + BLOCK_KM / 2) * 1000; y = (b.bn + BLOCK_KM / 2) * 1000
        lon[idx], lat[idx] = tr.transform(x.values, y.values)
    blocks["lon"], blocks["lat"] = lon, lat
    return blocks


def farthest_point_order(blocks: pd.DataFrame, rng) -> list[int]:
    xy = np.c_[blocks.lon * np.cos(np.radians(blocks.lat.mean())), blocks.lat]
    order = [int(rng.integers(len(blocks)))]
    d = np.linalg.norm(xy - xy[order[0]], axis=1)
    for _ in range(len(blocks) - 1):
        i = int(d.argmax())
        if d[i] == 0:
            break
        order.append(i)
        d = np.minimum(d, np.linalg.norm(xy - xy[i], axis=1))
    return order


def in_leaf_on(dt) -> bool:
    (m0, d0), (m1, d1) = LEAF_ON
    return date(dt.year, m0, d0) <= dt.date() <= date(dt.year, m1, d1)


def block_tiles(df, row) -> pd.DataFrame:
    return df[(df.zone == row.zone) & (df.year == row.year)
              & (df.e_km // BLOCK_KM * BLOCK_KM == row.be)
              & (df.n_km // BLOCK_KM * BLOCK_KM == row.bn)]


def check_dates(tiles: pd.DataFrame):
    starts, ends = [], []
    for url in tiles.Download_LAZ:
        lo, hi = copc_time_range(url)
        if not (in_leaf_on(lo) and in_leaf_on(hi)):
            return None
        starts.append(lo); ends.append(hi)
    if (max(ends) - min(starts)).days > MAX_SPAN_DAYS:
        return None
    return min(starts), max(ends)


def main(n_sites: int = 12):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = parse_tiles()
    blocks = candidate_blocks(df)
    blocks.to_csv(OUT / "candidate_blocks.csv", index=False)
    print(f"{len(blocks)} eligible 5x5 km blocks (year >= {MIN_YEAR}, full CHM coverage)")
    print(blocks.groupby(["zone", "year"]).size().unstack(fill_value=0))

    # spread out first, then accept in that order if flight dates pass;
    # skip candidates too close (< 0.5 deg) to an accepted site
    chosen, rows = [], []
    for i in farthest_point_order(blocks, rng):
        if len(chosen) == n_sites:
            break
        b = blocks.loc[i]
        if any(np.hypot(b.lon - c.lon, b.lat - c.lat) < 0.5 for c in chosen):
            continue
        tiles = block_tiles(df, b)
        try:
            dates = check_dates(tiles)
        except Exception as e:  # network / header problems: skip the block
            print("skip", int(b.zone), int(b.be), int(b.bn), int(b.year), e)
            continue
        if dates is None:
            print(f"reject Z{int(b.zone)}/E{int(b.be)}/N{int(b.bn)}/{int(b.year)}: flights outside leaf-on window or span > 60 days")
            continue
        chosen.append(b)
        site_id = f"Z{int(b.zone)}_E{int(b.be):04d}_N{int(b.bn):04d}_{int(b.year)}"
        print(f"accept {site_id} lon={b.lon:.2f} lat={b.lat:.2f} flown {dates[0]:%Y-%m-%d}..{dates[1]:%Y-%m-%d}")
        rows.append(dict(site_id=site_id, zone=int(b.zone), year=int(b.year),
                         e0_m=int(b.be) * 1000, n0_m=int(b.bn) * 1000,
                         lon=b.lon, lat=b.lat,
                         flight_start=dates[0].isoformat(), flight_end=dates[1].isoformat(),
                         tiles=";".join(tiles.Tilename.sort_values())))
        tiles.assign(site_id=site_id).to_csv(OUT / f"tiles_{site_id}.csv", index=False)

    sites = pd.DataFrame(rows)
    sites.to_csv(OUT / "sites.csv", index=False)
    gpd.GeoDataFrame(sites.drop(columns="tiles"),
                     geometry=gpd.points_from_xy(sites.lon, sites.lat), crs="EPSG:4326"
                     ).to_file(OUT / "sites_points.geojson", driver="GeoJSON")
    print(f"\n{len(sites)} sites written to {OUT / 'sites.csv'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
