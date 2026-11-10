"""Final evaluation of the selected model on the Test set. Produces the figures and numbers for the write-up.

  - accuracy and macro-F1 with 95% bootstrap confidence intervals (all of Test, and Test without look-alikes)
  - per-class precision / recall and the confusion matrix
  - 'missed disaster' rate: how often a flood, fire or collapse is called Normal (the costly mistake)
  - calibration: are the confidence values trustworthy? (reliability diagram and ECE)
  - urgency score by true class, and how many images each score threshold flags
  - the most confident mistakes, and Grad-CAM heatmaps for correct and wrong predictions

    python scripts/08_final_evaluation.py --model outputs/mobilenetv2_224.keras
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronerescue import config as C
from dronerescue.metrics import plot_confusion, summarize
from dronerescue.model_utils import gradcam, load_model, model_input_size, overlay, predict_probs
from dronerescue.pipeline import load_split
from dronerescue.priority import DEFAULT_WEIGHTS, priority_score

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(C.OUTPUT_DIR / "mobilenetv2_224.keras"))
ap.add_argument("--data-dir", default=str(C.DATA_DIR))
ap.add_argument("--out-dir", default=str(C.OUTPUT_DIR))
ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
args = ap.parse_args()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(C.RANDOM_SEED)

model = load_model(args.model)
size = model_input_size(model)
x, y, exposed, _ = load_split(args.data_dir, size, "Test")
print(f"Model {Path(args.model).name} ({size}px) on {len(y):,} Test images ({int(exposed.sum())} with look-alikes)")
probs = predict_probs(model, x)
pred = probs.argmax(1)
conf = probs.max(1)
names = [C.DISPLAY_NAMES[c].split(" /")[0] for c in C.CLASSES]


# ------------------------------------------------------------ bootstrap confidence intervals
def boot_ci(yt, yp, n):
    accs, f1s = [], []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        accs.append((yt[i] == yp[i]).mean())
        f1s.append(f1_score(yt[i], yp[i], average="macro", labels=list(range(len(C.CLASSES))), zero_division=0))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"accuracy_ci95": q(accs), "macro_f1_ci95": q(f1s)}


res = {"model": Path(args.model).name, "img_size": size, "n_test": int(len(y)), "n_unexposed": int((~exposed).sum())}
for key, mask in (("test_all", np.ones(len(y), bool)), ("test_unexposed", ~exposed)):
    r = summarize(y[mask], pred[mask])
    r.update(boot_ci(y[mask], pred[mask], args.boot))
    res[key] = r
    print(f"{key:<15} accuracy {r['accuracy']:.4f} {np.round(r['accuracy_ci95'], 4).tolist()}   "
          f"macro-F1 {r['macro_f1']:.4f} {np.round(r['macro_f1_ci95'], 4).tolist()}   min-class recall {r['min_class_recall']:.3f}")

# ------------------------------------------------------------ the costly mistake: a disaster called Normal
normal = C.CLASS_TO_IDX["Normal"]
is_dis, pred_dis = y != normal, pred != normal
missed = float((is_dis & ~pred_dis).sum() / is_dis.sum())
false_alarm = float((~is_dis & pred_dis).sum() / (~is_dis).sum())
res["disaster_vs_normal"] = {"missed_disaster_rate": missed, "false_alarm_rate": false_alarm,
                             "n_disaster": int(is_dis.sum()), "n_normal": int((~is_dis).sum())}
print(f"A real disaster called Normal: {100 * missed:.1f}%   A normal scene flagged as disaster: {100 * false_alarm:.1f}%")

# ------------------------------------------------------------ calibration
bins = np.linspace(0.25, 1.0, 11)
idx = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
rows, ece = [], 0.0
for b in range(len(bins) - 1):
    m = idx == b
    if m.sum() == 0:
        continue
    acc_b, conf_b = float((pred[m] == y[m]).mean()), float(conf[m].mean())
    ece += m.mean() * abs(acc_b - conf_b)
    rows.append((conf_b, acc_b, int(m.sum())))
res["calibration"] = {"ece": float(ece), "mean_confidence": float(conf.mean()),
                      "low_confidence_share_below_0.6": float((conf < 0.6).mean()),
                      "accuracy_when_confidence_below_0.6": float((pred[conf < 0.6] == y[conf < 0.6]).mean()) if (conf < 0.6).any() else None}
print(f"Calibration: ECE {ece:.3f}, mean confidence {conf.mean():.3f}, accuracy {(pred == y).mean():.3f}")

fig, ax = plt.subplots(figsize=(5.4, 5))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", color="#2ee6be", label="model")
for cb, ab, n in rows:
    ax.text(cb, ab - 0.05, str(n), fontsize=7, ha="center", color="gray")
ax.set_xlabel("confidence"); ax.set_ylabel("accuracy in that confidence range")
ax.set_title(f"Reliability diagram (ECE = {ece:.3f})\nnumbers = images per bin")
ax.set_xlim(0.2, 1.02); ax.set_ylim(0, 1.05); ax.legend(frameon=False); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(out / "final_reliability.png", dpi=150); plt.close()

# ------------------------------------------------------------ confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
plot_confusion(axes[0], res["test_all"]["confusion_matrix"], f"Test (all, {len(y)} images)")
plot_confusion(axes[1], res["test_unexposed"]["confusion_matrix"], f"Test without look-alikes ({int((~exposed).sum())})")
plt.tight_layout(); plt.savefig(out / "final_confusion.png", dpi=150); plt.close()

# ------------------------------------------------------------ urgency score
score = np.array([priority_score(p, DEFAULT_WEIGHTS) for p in probs])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.boxplot([score[y == k] for k in range(len(C.CLASSES))], tick_labels=names, patch_artist=True,
           boxprops=dict(facecolor="#2ee6be55"), medianprops=dict(color="#ff6b4a"))
for thr, lab in ((70, "High"), (40, "Medium")):
    ax.axhline(thr, color="gray", ls="--", lw=1); ax.text(4.45, thr + 1, lab, fontsize=8, color="gray", ha="right")
ax.set_ylabel("urgency score (0-100)"); ax.set_title("Urgency score by true class (default weights)")
plt.tight_layout(); plt.savefig(out / "final_priority_by_class.png", dpi=150); plt.close()
res["urgency"] = {"flagged_ge_40": {C.CLASSES[k]: float((score[y == k] >= 40).mean()) for k in range(len(C.CLASSES))},
                  "flagged_ge_70": {C.CLASSES[k]: float((score[y == k] >= 70).mean()) for k in range(len(C.CLASSES))}}
print("Share of images with urgency >= 40, by true class:", {k: round(v, 3) for k, v in res["urgency"]["flagged_ge_40"].items()})

# ------------------------------------------------------------ confident mistakes and Grad-CAM
wrong = np.where(pred != y)[0]
worst = wrong[np.argsort(-conf[wrong])][:12]
fig, axes = plt.subplots(3, 4, figsize=(11, 9.5))
for ax in axes.ravel():
    ax.axis("off")
for ax, i in zip(axes.ravel(), worst):
    ax.imshow(x[i])
    ax.set_title(f"true: {names[y[i]]}\npredicted: {names[pred[i]]} ({conf[i]:.2f})", fontsize=9, color="#c0392b")
plt.suptitle("Most confident mistakes"); plt.tight_layout(); plt.subplots_adjust(hspace=0.45)
plt.savefig(out / "final_confident_mistakes.png", dpi=130); plt.close()

right = np.where(pred == y)[0]
pick = list(rng.choice(right, size=min(4, len(right)), replace=False)) + list(worst[:4])
try:
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.4))
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(overlay(x[i], gradcam(model, x[i].astype(np.float32), int(pred[i]))))
        ok = pred[i] == y[i]
        ax.set_title(f"{names[y[i]]} -> {names[pred[i]]} ({conf[i]:.2f})", fontsize=9, color="#1e8449" if ok else "#c0392b")
        ax.axis("off")
    plt.suptitle("Grad-CAM: top row correct predictions, bottom row confident mistakes (warm = what drove the prediction)")
    plt.tight_layout(); plt.savefig(out / "final_gradcam.png", dpi=130); plt.close()
except Exception as e:                         # noqa: BLE001 - never lose the other results over a plotting problem
    print("Grad-CAM figure skipped:", e)

(out / "final_eval.json").write_text(json.dumps(res, indent=2))

a, u = res["test_all"], res["test_unexposed"]
md = f"""# Final evaluation: {res['model']}

Test set: {res['n_test']} images ({res['n_unexposed']} without a look-alike in the original Train/Val). 95% bootstrap intervals, {args.boot} resamples.

| Evaluation set | Accuracy | Macro-F1 | Weakest-class recall |
|---|---|---|---|
| All of Test | {a['accuracy']:.3f} ({a['accuracy_ci95'][0]:.3f} to {a['accuracy_ci95'][1]:.3f}) | {a['macro_f1']:.3f} ({a['macro_f1_ci95'][0]:.3f} to {a['macro_f1_ci95'][1]:.3f}) | {a['min_class_recall']:.3f} |
| Test without look-alikes | {u['accuracy']:.3f} ({u['accuracy_ci95'][0]:.3f} to {u['accuracy_ci95'][1]:.3f}) | {u['macro_f1']:.3f} ({u['macro_f1_ci95'][0]:.3f} to {u['macro_f1_ci95'][1]:.3f}) | {u['min_class_recall']:.3f} |

## Per-class results (all of Test)

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
""" + "\n".join(f"| {C.DISPLAY_NAMES[c]} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
                for c, v in a["per_class"].items()) + f"""

## The costly mistake

- A real disaster (flood, fire or collapse) called **Normal**: {100 * missed:.1f}% of {int(is_dis.sum())} disaster images.
- A normal scene flagged as a disaster: {100 * false_alarm:.1f}% of {int((~is_dis).sum())} normal images.

## Calibration

Mean confidence {conf.mean():.3f} against accuracy {(pred == y).mean():.3f}; expected calibration error {ece:.3f}.
{100 * (conf < 0.6).mean():.1f}% of images get a confidence below 0.6 (the app flags these for a manual check).

## Urgency score (default weights)

Share of images with a score of at least 40, by true class: """ + ", ".join(f"{C.DISPLAY_NAMES[c]} {100 * v:.0f}%" for c, v in res["urgency"]["flagged_ge_40"].items()) + "\n"
(out / "final_eval.md").write_text(md)
print(f"\nSaved final_eval.json/.md and figures to {out}")
