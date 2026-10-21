"""Run every verification check in order and print a pass/fail summary.

Usage: python -I verify/run_all.py [--quick]
  --quick  fewer cells in the slow label recompute (60 per site instead of 300)
Needs the data built by the pipeline (see README.md, "Reproduce").
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-I"]
R = ROOT / "verify/results"


def run(args):
    print(f"\n=== {' '.join(args)}", flush=True)
    subprocess.run(PY + [str(ROOT / args[0]), *args[1:]], check=True)


def main(quick):
    n = "60" if quick else "300"
    run(["verify/check_inputs.py"])
    run(["verify/check_labels.py", n])
    run(["verify/check_labels.py", "60", "1"])
    run(["verify/check_alignment.py"])
    run(["verify/check_splits.py"])
    run(["verify/check_metrics.py"])
    run(["verify/check_benchmarks.py"])
    run(["verify/check_baselines.py"])
    print("\n=== pytest", flush=True)
    pt = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], capture_output=True, text=True)
    print(pt.stdout[-400:])
    m = re.search(r"(\d+) passed", pt.stdout)
    n_tests = int(m.group(1)) if m else 0

    j = lambda n: json.loads((R / n).read_text())  # noqa: E731
    lab, ctl, al, sp, me, bm, inp = (j("labels.json"), j("labels_shift1m.json"), j("alignment.json"),
                                    j("splits.json"), j("metrics.json"), j("benchmarks.json"), j("inputs.json"))
    checks = [
        ("inputs: grids match, reflectance plausible", not inp["problems"]),
        ("labels: independent recompute exact", lab["summary"]["exact"] == 1.0),
        ("labels: 1 m shift is detected (check has power)", ctl["summary"]["exact"] < 0.9),
        ("alignment: pooled NDVI correlation peaks at zero shift",
         max(max(r) for r in al["summary"]["mean_grid"]) == al["summary"]["mean_grid"][3][3]),
        ("splits: no overlap, no shared tiles", not sp["footprint_overlaps"] and not sp["shared_tiles"]),
        ("splits: training code isolates test", all(sp["code_checks"].values())),
        ("metrics: sklearn matches stored results", me["worst_diff"] < 1e-5 and me["all_n_match"]),
        ("benchmarks: NTEMS/SCANFI match source pixels (>=99%)",
         all(r["exact"] >= 0.99 for r in bm["summary"] if r["product"] != "meta_chm_v2")),
        ("benchmarks: Meta conversion within 0.01 on average",
         all(r["mean_abs_diff"] < 0.01 for r in bm["summary"] if r["product"] == "meta_chm_v2")),
        ("unit tests pass", pt.returncode == 0),
    ]
    (R / "summary.json").write_text(json.dumps(dict(checks=[dict(name=n, ok=bool(o)) for n, o in checks],
                                                    passed=sum(o for _, o in checks), total=len(checks),
                                                    n_tests=n_tests), indent=1))
    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in checks))
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main("--quick" in sys.argv)
