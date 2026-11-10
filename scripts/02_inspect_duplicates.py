"""Phase 0, step 3 (optional but recommended): how much does the duplicate threshold matter?

01_eda_dedupe.py decides that two images are 'the same' when their 64-bit hashes differ in at most 5 bits
and their 16x16 thumbnails are close. That choice removes a lot of data, so this script shows
  1. a sensitivity table: how many images each stricter setting would remove,
  2. how duplicates are spread over classes and splits,
  3. contact sheets of Train/Test pairs grouped by hash distance, so you can judge by eye whether
     the pairs the checker calls 'the same' really are the same scene.

Needs the hash_cache.npz and manifest.csv that 01_eda_dedupe.py wrote into the data folder.

    python scripts/02_inspect_duplicates.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronerescue import config as C
from dronerescue.dupes import (assign_status, assign_status_pairwise, candidate_pairs, pair_mad,
                               union_find_groups)

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(C.DATA_DIR))
ap.add_argument("--out-dir", default=str(C.OUTPUT_DIR))
ap.add_argument("--pairs-per-group", type=int, default=4, help="Wildfire pairs and other-class pairs shown per sheet")
args = ap.parse_args()
data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
root = data_dir / C.DATASET_DIRNAME

cache = np.load(data_dir / "hash_cache.npz")
manifest = pd.read_csv(data_dir / "manifest.csv")
df = pd.DataFrame({"path": cache["paths"]}).merge(manifest[["path", "split", "class"]], on="path", how="left")
df["status"] = "keep"
hashes = [int(h, 16) for h in cache["dhash"]]
thumbs = list(cache["thumbs"])
n = len(df)
print(f"{n:,} readable images loaded from the cache")

print("Finding candidate pairs (hash distance <= 5) ...")
cand, _ = candidate_pairs(hashes, 5)
pairs = list(cand)
dist = np.array([cand[p] for p in pairs])
mad = pair_mad(thumbs, pairs)
i_arr = np.array([p[0] for p in pairs])
j_arr = np.array([p[1] for p in pairs])
print(f"{len(pairs):,} candidate pairs")

# ---------------------------------------------------------------- 1. sensitivity table
rows = []
for d in (0, 2, 5):
    for m in (4, 8, 12):
        sel = np.where((dist <= d) & (mad <= m))[0]
        sel_pairs = [pairs[k] for k in sel]
        groups = union_find_groups(n, sel_pairs)
        old = df.copy()
        assign_status(old, groups, np.arange(n))                         # old: whole chains removed
        work = df.copy()
        stats = assign_status_pairwise(work, sel_pairs, [int(dist[k]) for k in sel], np.arange(n))   # new default
        tr, tr_old = work[work["split"] == "Train"], old[old["split"] == "Train"]
        kept, kept_old = tr[tr["status"] == "keep"], tr_old[tr_old["status"] == "keep"]
        row = {"max hash distance": d, "max thumb diff": m, "groups": len(groups),
               "Train kept (pairwise)": len(kept), "Train dropped % (pairwise)": round(100 * (1 - len(kept) / len(tr)), 1),
               "Train kept (group, old)": len(kept_old),
               "Test with look-alike": stats["test_exposed"],
               "Test with look-alike %": round(100 * stats["test_exposed"] / (work["split"] == "Test").sum(), 1)}
        for c in C.CLASSES:
            row[f"Train {c} (pairwise)"] = int((kept["class"] == c).sum())
        rows.append(row)
sens = pd.DataFrame(rows)
sens["current setting"] = np.where((sens["max hash distance"] == C.HASH_MAX_DISTANCE) & (sens["max thumb diff"] == C.THUMB_MAX_MAD), "<--", "")
sens.to_csv(out_dir / "duplicate_sensitivity.csv", index=False)
print("\nSensitivity: Train images kept under each setting (pairwise = new default, group = previous method)")
print(sens.to_string(index=False))

# ---------------------------------------------------------------- 2. spread over classes
sel = np.where((dist <= C.HASH_MAX_DISTANCE) & (mad <= C.THUMB_MAX_MAD))[0]
groups = union_find_groups(n, [pairs[k] for k in sel])
work = df.copy()
assign_status(work, groups, np.arange(n))
in_group = work["dup_group"] >= 0
share = (work.assign(in_group=in_group).groupby(["split", "class"])["in_group"].mean().unstack().reindex(index=C.SPLITS, columns=C.CLASSES) * 100).round(1)
print("\nShare of images that belong to a duplicate group (%)")
print(share.to_string())
share.to_csv(out_dir / "duplicate_share_by_class.csv")

# ---------------------------------------------------------------- 3. Train-Test pairs by distance
split_i = df["split"].to_numpy()[i_arr]
split_j = df["split"].to_numpy()[j_arr]
cls_i = df["class"].to_numpy()[i_arr]
cross = ((split_i == "Train") & (split_j == "Test")) | ((split_i == "Test") & (split_j == "Train"))
tiers = {"d0": (0, 0), "d1-2": (1, 2), "d3-5": (3, 5)}
print("\nTrain-Test candidate pairs by hash distance (passing / failing the thumbnail check)")
tier_rows = []
for c in C.CLASSES:
    r = {"class": c}
    for name, (lo, hi) in tiers.items():
        m = cross & (cls_i == c) & (dist >= lo) & (dist <= hi)
        r[f"{name} pass"] = int((m & (mad <= C.THUMB_MAX_MAD)).sum())
        r[f"{name} fail"] = int((m & (mad > C.THUMB_MAX_MAD)).sum())
    tier_rows.append(r)
tier_df = pd.DataFrame(tier_rows)
print(tier_df.to_string(index=False))
tier_df.to_csv(out_dir / "duplicate_pairs_by_distance.csv", index=False)

rng = np.random.default_rng(C.RANDOM_SEED)
k = args.pairs_per_group
for name, (lo, hi) in tiers.items():
    ok_pairs = np.where(cross & (dist >= lo) & (dist <= hi) & (mad <= C.THUMB_MAX_MAD))[0]
    if len(ok_pairs) == 0:
        continue
    is_fire = cls_i[ok_pairs] == "Wildfire"
    pick = list(rng.choice(ok_pairs[is_fire], size=min(k, int(is_fire.sum())), replace=False)) + \
           list(rng.choice(ok_pairs[~is_fire], size=min(k, int((~is_fire).sum())), replace=False))
    fig, axes = plt.subplots(2, len(pick), figsize=(2.2 * len(pick), 5.0), squeeze=False)
    for col, q in enumerate(pick):
        a, b = int(i_arr[q]), int(j_arr[q])
        if df.at[a, "split"] == "Test":
            a, b = b, a                                     # always show the Train image on top
        for row, idx in enumerate((a, b)):
            with Image.open(root / df.at[idx, "path"]) as im:
                axes[row, col].imshow(ImageOps.fit(im.convert("RGB"), (200, 200)))
            axes[row, col].axis("off")
        axes[0, col].set_title(f"{df.at[a, 'class']}\nhash dist {dist[q]}, thumb {mad[q]:.1f}", fontsize=8)
        axes[1, col].set_title("Test", fontsize=8)
        axes[0, col].text(0.02, 0.97, "Train", transform=axes[0, col].transAxes, fontsize=8, color="white",
                          va="top", bbox=dict(facecolor="black", alpha=0.5, pad=1.5))
    plt.suptitle(f"Train image (top) and the Test image the checker calls the same, hash distance {lo}" +
                 (f" to {hi}" if hi != lo else ""), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"dup_pairs_{name}.png", dpi=130)
    plt.close()
    print(f"saved dup_pairs_{name}.png")

print(f"\nFiles written to {out_dir}: duplicate_sensitivity.csv, duplicate_share_by_class.csv, "
      "duplicate_pairs_by_distance.csv and dup_pairs_*.png")
