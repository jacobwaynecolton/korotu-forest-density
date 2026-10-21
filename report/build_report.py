"""Build report/index.html from template.html + data/manifest/eda_summary.json.

Usage: python -I report/build_report.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "data/manifest/eda_summary.json").read_text())
data["outline"] = json.loads((ROOT / "report/ontario_outline.json").read_text())
for s in data["per_site"]:  # keep the page payload small
    s["mean_cover"] = round(s["mean_cover"], 4)
    s["water"] = round(s["water"], 4)
    s["shares"] = [round(v, 4) for v in s["shares"]]
data["eval"] = {sp: json.loads((ROOT / f"data/manifest/eval_{sp}.json").read_text()) for sp in ("val", "test")}
data["final"] = json.loads((ROOT / "data/manifest/final_results.json").read_text())
cfgs = [json.loads((ROOT / f"models/{m}/config.json").read_text()) for m in ("unet_r34", "unet_r34_s1", "unet_r34_s2")]
data["train_minutes"] = sum(c["minutes"] for c in cfgs) / len(cfgs)
data["logs"] = [[{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(ROOT / f"models/{m}/log.csv"))]
                for m in ("unet_r34", "unet_r34_s1", "unet_r34_s2")]
data["unc"] = json.loads((ROOT / "data/manifest/uncertainty_preview.json").read_text())
VR = ROOT / "verify/results"
vj = lambda n: json.loads((VR / n).read_text())  # noqa: E731
summary = vj("summary.json")
al = vj("alignment.json")
repro = json.loads((ROOT / "models/unet_r34_repro/config.json").read_text())
orig = json.loads((ROOT / "models/unet_r34_before_offset_fix/config.json").read_text())
inputs = vj("inputs.json")
data["verify"] = dict(
    labels=vj("labels.json"), labels_shift=vj("labels_shift1m.json"),
    alignment=dict(summary=al["summary"]), splits={k: v for k, v in vj("splits.json").items() if k != "nearest_train"},
    metrics={k: v for k, v in vj("metrics.json").items() if k != "rows"}, benchmarks=vj("benchmarks.json"),
    baselines=vj("baselines.json"), inputs=inputs, runs=vj("training_runs.json"),
    before={sp: json.loads((ROOT / f"data/manifest/eval_{sp}_before_offset_fix.json").read_text()) for sp in ("val", "test")},
    repro=dict(val_mae=repro["best_val_mae"], test_mae=repro["test_mae"],
               original_val_mae=orig["best_val_mae"], original_test_mae=orig["test_mae"]),
    passed=summary["passed"], total=summary["total"], n_tests=summary["n_tests"],
    season_outside=inputs["scenes_outside_jun1_sep15"])
REPO = "https://github.com/jacobwaynecolton/korotu-forest-density"
html = (ROOT / "report/template.html").read_text().replace("__DATA__", json.dumps(data))
html = html.replace("UNCSITE", data["unc"]["figure_site"]).replace("REPO_SHORT", REPO.split("//")[1]).replace("REPO", REPO)
(ROOT / "report/index.html").write_text(html)
print(f"wrote report/index.html ({len(html) / 1e3:.0f} kB)")
