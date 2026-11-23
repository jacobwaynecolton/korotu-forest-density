"""Web server of the canopy-cover map tool (Python standard library only).

Serves the single-page app (app/static/) and a small JSON API:
  POST /api/analyses                 {"bbox": [west, south, east, north], "max_scenes": 1-4, "year": null | 2025...}
  GET  /api/analyses                 recent analyses
  GET  /api/analyses/<id>            status, progress and, when done, the result metadata
  GET  /api/analyses/<id>/<file>     maps.u16, overlay.u16, rgb_<k>.jpg, canopy_cover.tif
  GET  /api/geocode?q=...            place search (OpenStreetMap Nominatim, at most one request per second)
Analyses run one at a time in a background thread (engine.run); the 5 models run in parallel processes.
Access needs a token: open /?token=<token> once, the server then sets a cookie. The token is in DATA/app/token.txt.
The server listens on 127.0.0.1 only; reach it through an SSH or VS Code port forward.
Usage: python -I app/server.py [--port 8765]
"""
import argparse
import http.cookies
import json
import mimetypes
import queue
import re
import secrets
import shutil
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import engine  # noqa: E402
from common import DATA  # noqa: E402  (pipeline/common.py, on the path through engine)

STATIC = APP / "static"
JOBS_DIR = DATA / "app" / "analyses"
TOKEN_FILE = DATA / "app" / "token.txt"
KEEP = 60                       # analyses kept on disk
FILES = re.compile(r"^(maps\.u16|overlay\.u16|rgb_[0-3]\.jpg|canopy_cover\.tif)$")
JOBS, LOCK, TODO = {}, threading.Lock(), queue.Queue()
NOMINATIM = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "canopy-cover-map/1.0 (research prototype)"
_last_geocode = [0.0]


def save_status(job):
    d = JOBS_DIR / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps(job))


def load_jobs():
    for st in sorted(JOBS_DIR.glob("*/status.json")):
        try:
            j = json.loads(st.read_text())
        except json.JSONDecodeError:
            continue
        if j.get("state") in ("queued", "running"):
            j.update(state="error", message="The server restarted before this analysis finished. Run it again.")
        JOBS[j["id"]] = j


def prune():
    done = sorted((j for j in JOBS.values() if j["state"] in ("done", "error")), key=lambda j: j["created"])
    for j in done[:-KEEP]:
        shutil.rmtree(JOBS_DIR / j["id"], ignore_errors=True)
        JOBS.pop(j["id"], None)


def worker():
    while True:
        jid = TODO.get()
        job = JOBS[jid]

        def progress(msg, p):
            with LOCK:
                job.update(message=msg, progress=round(p, 3))

        with LOCK:
            job.update(state="running", started=time.time(), message="Starting")
        try:
            res = engine.run(JOBS_DIR / jid, job["bbox"], job["max_scenes"], job["year"], progress)
            with LOCK:
                job.update(state="done", result=res, finished=time.time(), progress=1.0, message="Done")
        except engine.AnalysisError as e:
            with LOCK:
                job.update(state="error", message=str(e), finished=time.time())
        except Exception as e:  # unexpected: keep the trace in the log, give the user a short message
            traceback.print_exc()
            with LOCK:
                job.update(state="error", finished=time.time(),
                           message=f"The analysis failed ({type(e).__name__}: {str(e)[:200]}). Try again, or another area.")
        save_status(job)
        with LOCK:
            prune()


class Handler(BaseHTTPRequestHandler):
    server_version = "CanopyMap/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    # ---------- helpers ----------
    def authorised(self):
        c = http.cookies.SimpleCookie(self.headers.get("Cookie", ""))
        return "canopy_token" in c and secrets.compare_digest(c["canopy_token"].value, TOKEN)

    def send(self, code, body, ctype="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype=None, attachment=None):
        data = path.read_bytes()
        headers = {"Content-Disposition": f'attachment; filename="{attachment}"'} if attachment else None
        self.send(200, data, ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream", headers)

    def deny(self):
        self.send(401, "<p>Open the link with its access token (…/?token=…) once; this browser will then be remembered.</p>",
                  "text/html; charset=utf-8")

    # ---------- routes ----------
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/" and "token" in q:
            if secrets.compare_digest(q["token"][0], TOKEN):
                self.send(303, "", "text/plain", {"Location": "/",
                          "Set-Cookie": f"canopy_token={TOKEN}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000"})
            else:
                self.deny()
            return
        if not self.authorised():
            return self.deny()
        if u.path == "/":
            return self.send_file(STATIC / "index.html", "text/html; charset=utf-8")
        m = re.match(r"^/static/([\w.-]+)$", u.path)
        if m and (STATIC / m.group(1)).is_file():
            return self.send_file(STATIC / m.group(1))
        if u.path == "/api/analyses":
            with LOCK:
                items = sorted(JOBS.values(), key=lambda j: -j["created"])[:30]
                return self.send(200, [{k: j.get(k) for k in ("id", "state", "bbox", "created", "message")} |
                                       {"year": (j.get("result") or {}).get("year"), "label": j.get("label")} for j in items])
        m = re.match(r"^/api/analyses/([0-9a-f]{12})$", u.path)
        if m:
            with LOCK:
                j = JOBS.get(m.group(1))
                if not j:
                    return self.send(404, {"error": "unknown analysis"})
                out = dict(j)
                if j["state"] == "queued":
                    out["queue_position"] = 1 + sum(1 for x in JOBS.values() if x["state"] == "queued" and x["created"] < j["created"])
            return self.send(200, out)
        m = re.match(r"^/api/analyses/([0-9a-f]{12})/([\w.]+)$", u.path)
        if m and FILES.match(m.group(2)):
            p = JOBS_DIR / m.group(1) / m.group(2)
            if not p.is_file():
                return self.send(404, {"error": "not found"})
            if p.suffix == ".tif":
                return self.send_file(p, "image/tiff", f"canopy_cover_{m.group(1)}.tif")
            return self.send_file(p, "application/octet-stream" if p.suffix == ".u16" else None)
        if u.path == "/api/geocode":
            text = (q.get("q") or [""])[0].strip()[:200]
            if not text:
                return self.send(400, {"error": "empty query"})
            wait = 1.0 - (time.time() - _last_geocode[0])
            if wait > 0:
                time.sleep(wait)
            _last_geocode[0] = time.time()
            try:
                r = requests.get(NOMINATIM, params=dict(q=text, format="jsonv2", limit=6), timeout=15,
                                 headers={"User-Agent": USER_AGENT, "Accept-Language": "en"})
                r.raise_for_status()
                return self.send(200, [dict(name=x.get("display_name"), lat=float(x["lat"]), lon=float(x["lon"]),
                                            bbox=[float(v) for v in x.get("boundingbox", [])]) for x in r.json()])
            except (requests.RequestException, ValueError) as e:
                return self.send(502, {"error": f"The place search is unavailable ({type(e).__name__})."})
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.authorised():
            return self.deny()
        if urlparse(self.path).path != "/api/analyses":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 10000)))
            bbox = [float(v) for v in body["bbox"]]
            assert len(bbox) == 4
            max_scenes = int(body.get("max_scenes", 4))
            assert 1 <= max_scenes <= 4
            year = body.get("year")
            year = int(year) if year not in (None, "", "auto") else None
            assert year is None or 2017 <= year <= 2100
            label = str(body.get("label", ""))[:120]
        except (ValueError, KeyError, AssertionError, TypeError, json.JSONDecodeError):
            return self.send(400, {"error": "Invalid request: bbox must be [west, south, east, north] in degrees."})
        try:
            engine.check_bbox(bbox)
        except engine.AnalysisError as e:
            return self.send(400, {"error": str(e)})
        jid = uuid.uuid4().hex[:12]
        job = dict(id=jid, state="queued", bbox=bbox, max_scenes=max_scenes, year=year, label=label,
                   created=time.time(), progress=0.0, message="Waiting in the queue")
        with LOCK:
            JOBS[jid] = job
        save_status(job)
        TODO.put(jid)
        return self.send(202, {"id": jid})


def main():
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    if not TOKEN_FILE.exists():
        TOKEN_FILE.write_text(secrets.token_urlsafe(18))
        TOKEN_FILE.chmod(0o600)
    TOKEN = TOKEN_FILE.read_text().strip()
    load_jobs()
    for s in engine.SEEDS:                          # start the model processes and load the 5 networks now
        engine.pool().submit(engine.warm, s)
    threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Canopy cover map running: http://localhost:{args.port}/?token={TOKEN}", flush=True)
    srv.serve_forever()


TOKEN = ""

if __name__ == "__main__":
    main()
