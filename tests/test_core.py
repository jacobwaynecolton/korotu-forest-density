"""Unit tests for the pieces everything else depends on. Run: .venv/bin/python -m pytest -q tests"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from sklearn.metrics import f1_score, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate  # noqa: E402
import make_labels  # noqa: E402
import make_pairs  # noqa: E402
from select_sites import CSRS_UTM_EPSG  # noqa: E402


@pytest.mark.parametrize("zone", [15, 16, 17, 18])
def test_csrs_epsg_codes_are_the_right_utm_zones(zone):
    name = CRS.from_epsg(CSRS_UTM_EPSG[zone]).name
    assert name == f"NAD83(CSRS) / UTM zone {zone}N"


@pytest.mark.parametrize("zone", [15, 16, 17, 18])
def test_label_grid_uses_sentinel2_crs(zone):
    assert CRS.from_epsg(make_labels.s2_epsg(zone)).name == f"WGS 84 / UTM zone {zone}N"


def test_tile_codes_thresholds(tmp_path):
    h = np.array([[-5, 0, 1.99, 2.0], [2.01, 5.0, 5.01, 90], [-9999, np.nan, 30, 1]], np.float32)
    p = tmp_path / "chm.tif"
    with rasterio.open(p, "w", driver="GTiff", width=4, height=3, count=1, dtype="float32",
                       crs="EPSG:3159", transform=from_origin(0, 3, 1, 1), nodata=-9999) as d:
        d.write(h, 1)
    code, _ = make_labels.tile_codes(p)
    # 0 nodata (incl. < -2 m and > 80 m), 1 <= 2 m, 2 > 2 m, 3 > 5 m
    assert code.tolist() == [[0, 1, 1, 1], [2, 2, 3, 0], [0, 0, 3, 1]]


def test_cover_is_fraction_above_2m_per_400_pixels():
    # the block-mean used in build_site: 20 x 20 fine pixels -> one 10 m cell
    K = make_labels.K
    assert K == 20
    fine = np.ones((2 * K, 2 * K), np.uint8)
    fine[:K, :K][:10, :] = 2           # top-left cell: 10 of 20 rows above 2 m -> 0.5
    fine[:K, K:] = 3                   # top-right: all > 5 m -> 1.0
    shp = (2, K, 2, K)
    cover = (fine >= 2).reshape(shp).sum(axis=(1, 3)) / (fine > 0).reshape(shp).sum(axis=(1, 3))
    assert cover.tolist() == [[0.5, 1.0], [0.0, 0.0]]


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(5000); p = np.clip(y + rng.normal(0, 0.15, y.size), 0, 1)
    m = evaluate.metrics(y, p)
    assert m["mae"] == pytest.approx(mean_absolute_error(y, p))
    assert m["r2"] == pytest.approx(r2_score(y, p))
    yc, pc = np.digitize(y, evaluate.CUTS), np.digitize(p, evaluate.CUTS)
    assert m["f1_macro"] == pytest.approx(f1_score(yc, pc, average="macro"))


def test_class_cuts_put_boundaries_in_the_upper_class():
    assert np.digitize([0.0, 0.099, 0.1, 0.4, 0.7, 1.0], evaluate.CUTS).tolist() == [0, 0, 1, 2, 3, 3]


def test_block3_mean_and_missing():
    a = np.arange(36, dtype=float).reshape(6, 6)
    a[0, 0] = np.nan
    b = evaluate.block3(a)
    assert np.isnan(b[0, 0])                      # any missing cell -> missing 30 m block
    assert b[1, 1] == pytest.approx(a[3:6, 3:6].mean())


def _item(flag, offset=-0.1):
    rb = [{"nodata": 0, "scale": 1e-4, "offset": offset}]
    return SimpleNamespace(properties={"earthsearch:boa_offset_applied": flag},
                           assets={"blue": SimpleNamespace(href="x", extra_fields={"raster:bands": rb})})


@pytest.mark.parametrize("flag,dn,expected", [
    (True, 120, True),      # flag says applied
    (False, 1150, False),   # offset still in the pixels: dark forest blue ~0.015 -> DN 1150
    (False, 150, True),     # flag wrong: pixels already offset-free (the bug the checks caught)
])
def test_boa_offset_detection(monkeypatch, flag, dn, expected):
    monkeypatch.setattr(make_pairs, "read_window", lambda *a, **k: np.full((10, 10), dn, np.uint16))
    assert make_pairs.boa_offset_applied(_item(flag), None, (10, 10)) is expected
