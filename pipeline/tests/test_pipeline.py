"""Unit tests of the final pipeline's core rules. Run: .venv/bin/python -m pytest -q pipeline/tests"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lidar  # noqa: E402
import model  # noqa: E402
import sentinel2  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_block_grid_matches_an_existing_label_grid():
    # Z17_E0600_N5175_2020: label grid of the first version, built from the CHM files
    assert common.block_grid(17, 600, 5175) == (600000, 5175060, 604980, 5179980)


def test_block_grid_is_snapped_and_inside_the_block():
    l, b, r, t = common.block_grid(15, 545, 5470)
    assert all(v % 60 == 0 for v in (l, b, r, t))
    assert 4900 <= r - l <= 5000 and 4900 <= t - b <= 5000


def test_site_id_round_trip():
    assert common.parse_site_id(common.site_id(16, 470, 5480, 2021)) == (16, 470, 5480, 2021)


def test_undeclared_nodata_conventions_are_missing(tmp_path):
    a = np.array([[1.0, 3.0, 6.0, -9999.0],
                  [np.finfo(np.float32).tiny, -32767.0, 85.0, 0.0]], np.float32)
    p = tmp_path / "t.tif"
    with rasterio.open(p, "w", driver="GTiff", width=4, height=2, count=1, dtype="float32", crs="EPSG:2958",
                       transform=from_origin(0, 2, 1, 1)) as d:   # no nodata declared
        d.write(a, 1)
    code, missing, _ = lidar.tile_codes(p)
    assert missing == 4                                  # -9999, FLT_MIN, -32767, 85 m
    assert code.tolist() == [[1, 2, 3, 0], [0, 0, 0, 1]]


def test_label_is_share_of_chm_pixels_above_2m(tmp_path):
    # one CSRS tile, 100 x 100 m at 0.5 m, a 1 m checkerboard with one pixel in four above 2 m
    h = np.zeros((200, 200), np.float32)
    h[::2, ::2] = 12.0
    p = tmp_path / "1kmZ17_CHM.tif"
    with rasterio.open(p, "w", driver="GTiff", width=200, height=200, count=1, dtype="float32", crs="EPSG:2958",
                       transform=from_origin(600000, 5175100, 0.5, 0.5), nodata=-9999.0) as d:
        d.write(h, 1)
    label, stats = lidar.build_label(17, (600020, 5175020, 600060, 5175060), [p])
    assert label.shape == (3, 4, 4)
    assert np.allclose(label[0], 0.25, atol=0.01) and np.allclose(label[1], 0.25, atol=0.01)
    assert np.all(label[2] == 1) and stats["cells"] == 16 and stats["partial_share"] == 1.0


def test_offset_alarm():
    class Item:
        id = "x"
        assets = {key: type("A", (), {"extra_fields": {"raster:bands": [{"scale": 0.0001}]}})()
                  for _, key, _ in sentinel2.BANDS}
    sentinel2.check_scale_and_offset(Item, np.array([90, 300, 900, 2000], np.uint16))   # B12 of a smoky scene too
    with pytest.raises(ValueError):
        sentinel2.check_scale_and_offset(Item, np.full(100, 1100, np.uint16))


def test_reflectance_and_mask():
    x = np.zeros((13, 2, 2), np.uint16)
    x[:12] = 500
    x[0, 0, 0] = 0                                       # no data in one band
    x[12] = [[4, 9], [3, 5]]                             # vegetation, cloud high, shadow, bare
    refl, bad = sentinel2.reflectance_and_mask(x)
    assert np.isnan(refl[0, 0, 0]) and refl[1, 0, 0] == pytest.approx(0.05)
    assert bad.tolist() == [[False, True], [True, False]]


def test_mixture_of_identical_seeds_gives_them_back():
    q = np.array([[0.3], [0.5], [0.8]])
    assert np.allclose(model.mixture([q] * 5), q)
    assert np.allclose(model.mixture([q]), q)


def test_per_band_calibration_reaches_target_coverage():
    rng = np.random.default_rng(0)
    n = 200_000
    q50 = rng.uniform(0, 1, n)
    y = np.clip(q50 + rng.normal(0, 0.02 + 0.25 * q50 * (1 - q50)), 0, 1)
    q10, q90 = np.clip(q50 - 0.05, 0, 1), np.clip(q50 + 0.05, 0, 1)
    a, b = slice(0, n // 2), slice(n // 2, n)
    lo, hi = model.calibrate(q10[b], q50[b], q90[b], model.constants(y[a], q10[a], q50[a], q90[a]))
    band = model.band_of(q50[b], 10)
    for i in range(2, 8):
        m = band == i
        assert 0.76 < ((y[b][m] >= lo[m]) & (y[b][m] <= hi[m])).mean() < 0.84


def test_quotas_sum_to_target():
    sel = load("02_select_sites.py")
    for n in (5, 250, 251):
        assert sum(sel.quotas(n).values()) == n


class FirstBandModel(torch.nn.Module):
    """Logits whose q50 equals the first input band: each date predicts its own value."""
    def forward(self, x):
        v = x[:, :1].clamp(1e-4, 1 - 1e-4)
        return torch.cat([torch.log(v / (1 - v)), x[:, :1] * 0, x[:, :1] * 0], 1)


def test_predict_averages_unflagged_dates_only():
    x = torch.zeros(2, 12, 8, 8)
    x[0, 0], x[1, 0] = 0.2, 0.6
    bad = torch.zeros(2, 8, 8, dtype=torch.bool)
    bad[1, :4] = True                                    # second date flagged on the top half
    q = model.predict(FirstBandModel(), x, bad, tile=8, overlap=0)
    assert np.allclose(q[1, :4], 0.2, atol=1e-3) and np.allclose(q[1, 4:], 0.4, atol=1e-3)
