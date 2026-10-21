"""List every training run (from models/*/config.json) for the report's run log.

The 60-step smoke test's model folder was deleted; its scores come from the console log.
Writes verify/results/training_runs.json.
Usage: python -I verify/list_runs.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHY = {
    "unet_r34_before_offset_fix": ("First full run, seed 0 (before the offset fix)", "Superseded"),
    "unet_r34_s1_before_offset_fix": ("Seed 1, same settings (before the offset fix)", "Superseded"),
    "unet_r34_s2_before_offset_fix": ("Seed 2, same settings (before the offset fix)", "Superseded"),
    "unet_r34_repro": ("From-scratch rerun of seed 0 on the pre-fix data, to test reproducibility", "Check only"),
    "unet_r34": ("Seed 0 after the offset fix", "Ensemble"),
    "unet_r34_s1": ("Seed 1 after the offset fix", "Ensemble"),
    "unet_r34_s2": ("Seed 2 after the offset fix", "Ensemble"),
}


def main():
    runs = [dict(name="smoke", why="60-step pipeline smoke test (model not kept)", steps=60, seed=0,
                 val_mae=0.1506, test_mae=0.1586, used="No")]
    for name, (why, used) in WHY.items():
        c = json.loads((ROOT / "models" / name / "config.json").read_text())
        runs.append(dict(name=name, why=why, steps=c["steps"], seed=c["seed"], val_mae=c["best_val_mae"],
                         test_mae=c["test_mae"], used=used, minutes=c["minutes"]))
    out = ROOT / "verify/results/training_runs.json"
    out.write_text(json.dumps(runs, indent=1))
    for r in runs:
        print(f"{r['name']:32s} val {r['val_mae']:.4f} test {r['test_mae']:.4f}  {r['used']}")


if __name__ == "__main__":
    main()
