"""Phase 1, step 2: classical baseline on hand-made image features.

Features (63 numbers per image): mean and spread of each colour channel, colour histograms (hue, saturation,
brightness), edge strength and direction histogram, and the mean colour of each quarter of the image
(sky/ground layout). Models: logistic regression and a random forest, both with class weights.
The model is chosen by validation macro-F1. Test is only used for the final report.

    python scripts/04_baseline.py --size 128
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronerescue import config as C
from dronerescue.metrics import evaluate_all, plot_confusion
from dronerescue.pipeline import load_split

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(C.DATA_DIR))
ap.add_argument("--out-dir", default=str(C.OUTPUT_DIR))
ap.add_argument("--size", type=int, default=C.DEFAULT_IMG_SIZE)
args = ap.parse_args()
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)


def hist_counts(idx, n_bins):
    """Per-image histogram of integer bin indices, shape (N, n_bins), without a Python loop over images."""
    n = idx.shape[0]
    flat = (idx.reshape(n, -1) + n_bins * np.arange(n)[:, None]).ravel()
    return np.bincount(flat, minlength=n * n_bins).reshape(n, n_bins) / idx[0].size


def extract_features(x, chunk=256):
    out = []
    for s in range(0, len(x), chunk):
        b = x[s:s + chunk].astype(np.float32) / 255.0
        n, h, w, _ = b.shape
        feats = [b.mean(axis=(1, 2)), b.std(axis=(1, 2))]                       # 3 + 3
        hsv = rgb_to_hsv(b)
        for ch, bins in enumerate((18, 8, 8)):                                  # hue, saturation, brightness
            feats.append(hist_counts(np.clip((hsv[..., ch] * bins).astype(int), 0, bins - 1), bins))
        gray = b.mean(axis=3)
        gx, gy = np.diff(gray, axis=2)[:, :-1, :], np.diff(gray, axis=1)[:, :, :-1]
        mag, ang = np.hypot(gx, gy), np.arctan2(gy, gx)
        feats.append(np.stack([mag.mean(axis=(1, 2)), mag.std(axis=(1, 2)), (mag > 0.1).mean(axis=(1, 2))], axis=1))
        obin = np.clip(((ang + np.pi) / (2 * np.pi) * 8).astype(int), 0, 7)    # orientation histogram weighted by strength
        flat = (obin.reshape(n, -1) + 8 * np.arange(n)[:, None]).ravel()
        oh = np.bincount(flat, weights=mag.reshape(n, -1).ravel(), minlength=n * 8).reshape(n, 8)
        feats.append(oh / np.maximum(oh.sum(axis=1, keepdims=True), 1e-8))
        q = b.reshape(n, 2, h // 2, 2, w // 2, 3).mean(axis=(2, 4)).reshape(n, -1)   # 2x2 grid of mean colours
        feats.append(q)
        out.append(np.concatenate(feats, axis=1))
    return np.concatenate(out).astype(np.float32)


t0 = time.time()
splits = {s: load_split(args.data_dir, args.size, s) for s in C.SPLITS}
print({s: v[0].shape for s, v in splits.items()})
F = {s: extract_features(v[0]) for s, v in splits.items()}
print(f"features: {F['Train'].shape[1]} per image  ({time.time() - t0:.0f} s)")
y_tr, y_val, y_te = splits["Train"][1], splits["Val"][1], splits["Test"][1]
exposed = splits["Test"][2]

models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    "Random forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=C.RANDOM_SEED),
}
results = {}
for name, model in models.items():
    t1 = time.time()
    model.fit(F["Train"], y_tr)
    res = evaluate_all(y_val, model.predict(F["Val"]), y_te, model.predict(F["Test"]), exposed)
    res["train_seconds"] = round(time.time() - t1, 1)
    results[name] = res
    print(f"{name:<20} val macro-F1 {res['val']['macro_f1']:.3f} | test acc {res['test_all']['accuracy']:.3f} "
          f"macro-F1 {res['test_all']['macro_f1']:.3f} | unexposed macro-F1 "
          f"{res.get('test_unexposed', {}).get('macro_f1', float('nan')):.3f}")

best = max(results, key=lambda k: results[k]["val"]["macro_f1"])      # chosen on validation only
print(f"Best on validation macro-F1: {best}")
(out_dir / "phase1_baseline_results.json").write_text(json.dumps(
    {"models": results, "selected_by_validation": best, "n_features": int(F["Train"].shape[1]),
     "img_size": args.size, "demo_data": False}, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
plot_confusion(axes[0], results[best]["test_all"]["confusion_matrix"], f"{best}: Test (all)")
if "test_unexposed" in results[best]:
    plot_confusion(axes[1], results[best]["test_unexposed"]["confusion_matrix"], f"{best}: Test without look-alikes")
plt.tight_layout()
plt.savefig(out_dir / "phase1_baseline_confusion.png", dpi=150)

if hasattr(models["Random forest"], "feature_importances_"):
    names = (["mean R", "mean G", "mean B", "std R", "std G", "std B"] + [f"hue{i}" for i in range(18)] +
             [f"sat{i}" for i in range(8)] + [f"val{i}" for i in range(8)] + ["edge mean", "edge std", "edge share"] +
             [f"orient{i}" for i in range(8)] + [f"quad{q}{c}" for q in range(4) for c in "RGB"])
    imp = models["Random forest"].feature_importances_
    top = np.argsort(imp)[::-1][:12]
    plt.figure(figsize=(6.5, 4))
    plt.barh([names[i] for i in top][::-1], imp[top][::-1], color="#2ee6be")
    plt.title("Random forest: most useful features")
    plt.tight_layout()
    plt.savefig(out_dir / "phase1_baseline_features.png", dpi=150)
print(f"Saved results to {out_dir}")
