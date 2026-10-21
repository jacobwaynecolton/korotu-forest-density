"""Read flight dates from FRI COPC .laz headers via HTTP range requests (no full download).

COPC files put a 'copc info' VLR right after the LAS 1.4 header; it stores the
min/max GPS time of all points. LAS 1.4 uses adjusted standard GPS time
(GPS seconds since 1980-01-06 minus 1e9).
"""
import struct, sys
from datetime import datetime, timedelta, timezone
import requests

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
LEAP_SECONDS = 18  # GPS-UTC offset since 2017


def gps_to_utc(adjusted_gps: float) -> datetime:
    return GPS_EPOCH + timedelta(seconds=adjusted_gps + 1e9 - LEAP_SECONDS)


def copc_time_range(url: str, timeout: int = 60):
    with requests.get(url, headers={"Range": "bytes=0-1023"}, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        b = r.raw.read(1024)  # server may ignore Range; never pull the whole file
    if b[:4] != b"LASF":
        raise ValueError(f"not a LAS file: {url}")
    global_encoding = struct.unpack_from("<H", b, 6)[0]
    header_size = struct.unpack_from("<H", b, 94)[0]
    user_id = b[header_size + 2:header_size + 18].rstrip(b"\0")
    if user_id != b"copc":
        raise ValueError(f"first VLR is {user_id!r}, not copc info")
    info = header_size + 54  # VLR header is 54 bytes
    gmin, gmax = struct.unpack_from("<dd", b, info + 56)
    # FRI files leave global-encoding bit 0 unset but still store adjusted
    # standard GPS time; week-seconds would never exceed 604800.
    if not (global_encoding & 1) and gmax <= 604800:
        raise ValueError("GPS week time; date needs the GPS week number")
    return gps_to_utc(gmin), gps_to_utc(gmax)


if __name__ == "__main__":
    for u in sys.argv[1:]:
        lo, hi = copc_time_range(u)
        print(u.rsplit("/", 1)[-1], lo.isoformat(), hi.isoformat())
