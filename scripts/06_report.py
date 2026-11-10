"""Put every trained model side by side.

Reads the results JSON files written by 04_baseline.py, 05_cnn_scratch.py and 07_transfer_mobilenetv2.py and writes
outputs/model_comparison.md and outputs/model_comparison.png.

    python scripts/06_phase1_report.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronerescue import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default=str(C.OUTPUT_DIR))
args = ap.parse_args()
out = Path(args.out_dir)

entries = {}
base_f = out / "phase1_baseline_results.json"
if base_f.exists():
    b = json.loads(base_f.read_text())
    for name, r in b["models"].items():
        entries[f"{name} (features)"] = r
    best_baseline = f"{b['selected_by_validation']} (features)"
for f in sorted(list(out.glob("phase1_cnn_*_results.json")) + list(out.glob("phase2_*_results.json"))):
    entries[f.stem.replace("phase1_", "").replace("phase2_", "").replace("_results", "")] = json.loads(f.read_text())
if not entries:
    sys.exit("No results found. Run 04_baseline.py, 05_cnn_scratch.py or 07_transfer_mobilenetv2.py first.")


def get(r, part, key):
    return r.get(part, {}).get(key, float("nan"))


lines = ["# Model comparison", "",
         "Models are chosen on the validation set only. 'Unexposed' means Test images without a look-alike in Train/Val.", "",
         "| Model | Val macro-F1 | Test accuracy | Test macro-F1 | Test min-class recall | Unexposed macro-F1 | Gap (all - unexposed) |",
         "|---|---|---|---|---|---|---|"]
for name, r in entries.items():
    a, u = get(r, "test_all", "macro_f1"), get(r, "test_unexposed", "macro_f1")
    lines.append(f"| {name} | {get(r, 'val', 'macro_f1'):.3f} | {get(r, 'test_all', 'accuracy'):.3f} | {a:.3f} | "
                 f"{get(r, 'test_all', 'min_class_recall'):.3f} | {u:.3f} | {a - u:+.3f} |")
lines += ["", "## Per-class recall on Test (all / unexposed)", "",
          "| Model | " + " | ".join(C.DISPLAY_NAMES[c].split(" /")[0] for c in C.CLASSES) + " |",
          "|---|" + "---|" * len(C.CLASSES)]
for name, r in entries.items():
    cells = []
    for c in C.CLASSES:
        a = r["test_all"]["per_class"][c]["recall"]
        u = r.get("test_unexposed", {}).get("per_class", {}).get(c, {}).get("recall", float("nan"))
        cells.append(f"{a:.2f} / {u:.2f}")
    lines.append(f"| {name} | " + " | ".join(cells) + " |")
lines += ["", "Recall matters most here: a missed flood or collapse costs more than a false alarm.", ""]
(out / "model_comparison.md").write_text("\n".join(lines))
print("\n".join(lines))

names = list(entries)
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.2))
for k, (part, label, col) in enumerate([("val", "Val", "#8c6eff"), ("test_all", "Test (all)", "#2ee6be"),
                                         ("test_unexposed", "Test (no look-alikes)", "#f5b342")]):
    ax.bar(x + (k - 1) * 0.26, [get(entries[n], part, "macro_f1") for n in names], 0.26, label=label, color=col)
ax.set_xticks(x, names, rotation=15, ha="right")
ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1)
ax.set_title("macro-F1 by model and evaluation set")
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(out / "model_comparison.png", dpi=150)
