"""Phase 0, step 2: explore AIDERv2, check data quality and build the cleaned file list.

What it does
  1. Reads every image once: size, mean colour, corrupt-file check, and a perceptual hash.
  2. Finds exact and near-duplicate images, including copies shared between Train, Val and Test.
     (Leakage: a test image the model has already seen in training inflates the test score.)
  3. Applies the cleaning rules below and writes manifest.csv (every image, with a status)
     and manifest_clean.csv (only the images later phases will use).
  4. Saves plots and a written summary to outputs/.

Cleaning rules, default pairwise policy (Test is never changed, so results stay comparable with other work)
  - corrupt or tiny image                      -> dropped
  - Test image                                 -> always kept
  - Val image that also appears in Test        -> dropped from Val
  - Train image that resembles a Val/Test one  -> dropped from Train
  - exact copies inside Train or inside Val    -> keep the first, drop the rest
  - merely similar images inside one split     -> kept (use --policy group to remove whole chains)
  - same picture under different labels       -> dropped from Train/Val (which label is right is
                                                 unknown); kept in Test but flagged for review

    python scripts/01_eda_dedupe.py                       # real dataset in $DRONERESCUE_DATA
    python scripts/01_eda_dedupe.py --data-dir demo_data --out-dir outputs_demo
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronerescue import config as C
from dronerescue.data import list_images
from dronerescue.dupes import (analyze_image, assign_status, assign_status_pairwise, candidate_pairs,
                               pair_mad, union_find_groups)

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(C.DATA_DIR))
ap.add_argument("--out-dir", default=str(C.OUTPUT_DIR))
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--max-dist", type=int, default=C.HASH_MAX_DISTANCE, help="max differing hash bits (0-5)")
ap.add_argument("--max-mad", type=float, default=C.THUMB_MAX_MAD, help="max thumbnail difference (0-255 scale)")
ap.add_argument("--policy", choices=["pairwise", "group"], default="pairwise",
                help="pairwise: remove only direct look-alikes (default); group: remove whole chains of similar images")
args = ap.parse_args()
data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
root = data_dir / C.DATASET_DIRNAME


# ------------------------------------------------------------------ main
t0 = time.time()
rows = list_images(data_dir)
df = pd.DataFrame(rows, columns=["path", "split", "class"])
print(f"Found {len(df):,} image files in {root}")

print("Reading every image (size, colour, corrupt check, hash) ...")
results = []
with ProcessPoolExecutor(max_workers=args.workers) as ex:
    for k, r in enumerate(ex.map(partial(analyze_image, root), df["path"].tolist(), chunksize=32), 1):
        results.append(r)
        if k % 2000 == 0:
            print(f"  {k:,} / {len(df):,}")
info = pd.DataFrame(results)
df = df.merge(info.drop(columns=["thumb"]), on="path")
thumbs_by_path = {r["path"]: np.frombuffer(r["thumb"], dtype=np.uint8) for r in results if r["ok"]}

df["status"] = "keep"
df["dup_group"] = -1
df["label_conflict"] = False
df.loc[~df["ok"], "status"] = "corrupt"
small = df["ok"] & (df[["width", "height"]].min(axis=1) < C.MIN_SIDE_PX)
df.loc[small, "status"] = "too_small"

# --- duplicates among the readable images
valid = df[df["status"] == "keep"].copy()
v_idx = valid.index.to_numpy()
hashes = [int(h) for h in valid["dhash"]]
thumbs = [thumbs_by_path[p] for p in valid["path"]]
print("Searching for duplicates ...")
cand, skipped = candidate_pairs(hashes, args.max_dist)
cand_pairs = list(cand)
mads = pair_mad(thumbs, cand_pairs)
pairs = [pr for pr, m in zip(cand_pairs, mads) if m <= args.max_mad]
pair_dists = [cand[pr] for pr in pairs]
groups = union_find_groups(len(valid), pairs)
if skipped:
    print(f"  note: {skipped} hash bucket(s) were too large and only partly compared")
if args.policy == "pairwise":
    stats = assign_status_pairwise(df, pairs, pair_dists, v_idx)
else:
    stats = assign_status(df, groups, v_idx)
print(f"Policy: {args.policy}, max hash distance {args.max_dist}, max thumbnail difference {args.max_mad}")

# cache for scripts/02_inspect_duplicates.py (hashes and thumbnails of the readable images)
np.savez_compressed(data_dir / "hash_cache.npz", paths=np.array(valid["path"].tolist()),
                    dhash=np.array([f"{h:016x}" for h in hashes]), thumbs=np.stack(thumbs))

clean = df[df["status"] == "keep"]
df_out = df.drop(columns=["ok"]).copy()
df_out["dhash"] = df_out["dhash"].map(lambda v: "" if pd.isna(v) else f"{int(v):016x}")
df_out.to_csv(data_dir / "manifest.csv", index=False)
clean_cols = ["path", "split", "class", "width", "height", "label_conflict", "test_leak_exposed"]
df_out[df_out["status"] == "keep"][clean_cols].to_csv(data_dir / "manifest_clean.csv", index=False)
df_out[df_out["status"] != "keep"][["path", "split", "class", "status", "dup_group", "error"]].to_csv(
    out_dir / "dropped_files.csv", index=False)
df_out[df_out["dup_group"] >= 0][["dup_group", "path", "split", "class", "status", "label_conflict"]].sort_values(
    ["dup_group", "split"]).to_csv(out_dir / "duplicates_report.csv", index=False)
df_out[df_out["label_conflict"]][["dup_group", "path", "split", "class", "status"]].sort_values(
    "dup_group").to_csv(out_dir / "label_conflicts.csv", index=False)

# ------------------------------------------------------------------ summary numbers
before = df.groupby(["split", "class"]).size().unstack(fill_value=0).reindex(index=C.SPLITS, columns=C.CLASSES, fill_value=0)
after = clean.groupby(["split", "class"]).size().unstack(fill_value=0).reindex(index=C.SPLITS, columns=C.CLASSES, fill_value=0)
status_counts = df["status"].value_counts().to_dict()
ok = df[df["ok"]]
short_side = ok[["width", "height"]].min(axis=1)
total_images = len(df)
count_check = "skipped (demo or partial data)"
if total_images >= 5000:
    exp = pd.DataFrame(C.EXPECTED_COUNTS).T.reindex(index=C.SPLITS, columns=C.CLASSES)
    diff = (before - exp)
    count_check = "matches the published table" if (diff == 0).all().all() else \
        "DIFFERS from the published table:\n" + diff.to_string()

summary = {
    "settings": {"policy": args.policy, "max_hash_distance": args.max_dist, "max_thumb_diff": args.max_mad},
    "total_files": int(total_images),
    "status_counts": {k: int(v) for k, v in status_counts.items()},
    "duplicate_groups": len(groups),
    "leaks": stats,
    "test_images_with_trainval_lookalike": int(df["test_leak_exposed"].sum()),
    "label_conflict_images": int(df["label_conflict"].sum()),
    "label_conflict_groups": int(df.loc[df["label_conflict"], "dup_group"].nunique()),
    "before": before.to_dict("index"), "after": after.to_dict("index"),
    "published_count_check": count_check,
    "short_side_px": {"min": int(short_side.min()), "median": int(short_side.median()),
                      "max": int(short_side.max()), "share_below_224": round(float((short_side < 224).mean()), 4)},
    "seconds": round(time.time() - t0, 1),
}
(out_dir / "phase0_summary.json").write_text(json.dumps(summary, indent=2))


def md_table(frame):
    cols = ["split"] + list(frame.columns) + ["total"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s, r in frame.iterrows():
        lines.append(f"| {s} | " + " | ".join(str(int(v)) for v in r) + f" | {int(r.sum())} |")
    return "\n".join(lines)


md = f"""# Phase 0 findings

Dataset: AIDERv2 ({total_images:,} image files found). Published counts check: {count_check}.

Duplicate settings: policy `{args.policy}`, max hash distance {args.max_dist}, max thumbnail difference {args.max_mad}.

## Data quality

| Check | Result |
|---|---|
| Corrupt or unreadable files | {status_counts.get('corrupt', 0)} |
| Images with a side shorter than {C.MIN_SIDE_PX}px | {status_counts.get('too_small', 0)} |
| Duplicate groups (2+ near-identical images) | {len(groups)} |
| Train images that also appear in Test (leakage) | {stats['leak_train_in_test']} |
| Train images that also appear in Val | {stats['leak_train_in_val']} |
| Val images that also appear in Test | {stats['leak_val_in_test']} |
| **Test images with a look-alike in Train or Val** | **{stats['test_exposed']} of {int((df['split'] == 'Test').sum())}** ({100 * stats['test_exposed'] / max(int((df['split'] == 'Test').sum()), 1):.1f}%) |
| Extra copies inside Test (kept) | {stats['within_test_copies']} |
| Copies inside Train/Val removed | {status_counts.get('dup_within', 0)} |
| Images with conflicting class labels | {summary['label_conflict_images']} in {summary['label_conflict_groups']} groups ({status_counts.get('dup_conflict', 0)} dropped from Train/Val) |
| Shorter side below 224px | {100 * summary['short_side_px']['share_below_224']:.1f}% (median {summary['short_side_px']['median']}px) |

## Images per split and class

Before cleaning

{md_table(before)}

After cleaning (what the models will use)

{md_table(after)}

Test images flagged `test_leak_exposed` stay in Test; Phase 3 reports accuracy on all of Test and on the unexposed part.

Dropped files are listed in `dropped_files.csv`; duplicate groups in `duplicates_report.csv`; label conflicts in `label_conflicts.csv`.
"""
(out_dir / "phase0_summary.md").write_text(md)

# ------------------------------------------------------------------ plots
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
for ax, split in zip(axes, C.SPLITS):
    x = np.arange(len(C.CLASSES))
    cols = [C.CLASS_COLORS[c] for c in C.CLASSES]
    ax.bar(x, before.loc[split].values, color=cols, alpha=0.3, label="before cleaning")
    ax.bar(x, after.loc[split].values, color=cols, label="after cleaning", width=0.55)
    for xi, v in zip(x, after.loc[split].values):
        ax.text(xi, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [C.DISPLAY_NAMES[c].split(" /")[0] for c in C.CLASSES])
    ax.set_title(f"{split}")
axes[0].set_ylabel("images")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=2, frameon=False)
plt.suptitle("Class balance per split", x=0.35)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(out_dir / "eda_class_distribution.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6.4, 5))
samp = ok.sample(min(len(ok), 4000), random_state=C.RANDOM_SEED)
for c in C.CLASSES:
    s = samp[samp["class"] == c]
    ax.scatter(s["width"], s["height"], s=7, alpha=0.5, color=C.CLASS_COLORS[c], label=C.DISPLAY_NAMES[c])
ax.axvline(224, color="gray", ls="--", lw=1)
ax.axhline(224, color="gray", ls="--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("width (px)"); ax.set_ylabel("height (px)")
ax.set_title(f"Image sizes (dashed = 224px model input), {100 * summary['short_side_px']['share_below_224']:.1f}% smaller")
ax.legend(frameon=False, markerscale=2)
plt.tight_layout()
plt.savefig(out_dir / "eda_image_sizes.png", dpi=150)
plt.close()

means = ok.groupby("class")[["mean_r", "mean_g", "mean_b"]].mean().reindex(C.CLASSES)
fig, ax = plt.subplots(figsize=(6.4, 4))
x = np.arange(len(C.CLASSES))
for k, (col, name) in enumerate([("mean_r", "R"), ("mean_g", "G"), ("mean_b", "B")]):
    ax.bar(x + (k - 1) * 0.26, means[col].values, width=0.26, color=["#e5484d", "#30a46c", "#3e63dd"][k], label=name)
ax.set_xticks(x, [C.DISPLAY_NAMES[c].split(" /")[0] for c in C.CLASSES])
ax.set_ylabel("mean pixel value (0-255)")
ax.set_title("Average colour per class")
ax.legend(frameon=False, ncol=3)
plt.tight_layout()
plt.savefig(out_dir / "eda_mean_colour.png", dpi=150)
plt.close()

rng = np.random.default_rng(C.RANDOM_SEED)
n_cols = 6
fig, axes = plt.subplots(len(C.CLASSES), n_cols, figsize=(1.9 * n_cols, 2.0 * len(C.CLASSES)))
for r, c in enumerate(C.CLASSES):
    pool = clean[(clean["split"] == "Train") & (clean["class"] == c)]["path"].tolist()
    pick = rng.choice(pool, size=min(n_cols, len(pool)), replace=False) if pool else []
    for k in range(n_cols):
        ax = axes[r, k]
        ax.axis("off")
        if k < len(pick):
            with Image.open(root / pick[k]) as im:
                ax.imshow(ImageOps.fit(im.convert("RGB"), (200, 200)))
    axes[r, 0].set_title(C.DISPLAY_NAMES[c], loc="left", fontsize=10, color=C.CLASS_COLORS[c], fontweight="bold")
plt.suptitle("Random Train images per class (centre-cropped squares)")
plt.tight_layout()
plt.savefig(out_dir / "eda_samples.png", dpi=130)
plt.close()

# leakage evidence: a Train image next to its copy in Test or Val
leak_groups = []
for gid in sorted(df.loc[df["status"].isin(["dup_leak", "dup_leak_test"]), "dup_group"].unique()):
    g = df[df["dup_group"] == gid]
    hi = g[g["split"].isin(["Test", "Val"])]
    lo = g[g["status"].isin(["dup_leak", "dup_leak_test"])]
    if len(hi) and len(lo):
        leak_groups.append((lo.iloc[0], hi.iloc[0]))
if leak_groups:
    show = leak_groups[:4]
    fig, axes = plt.subplots(2, len(show), figsize=(3 * len(show), 6.2), squeeze=False)
    for k, (lo_r, hi_r) in enumerate(show):
        for row, rec in enumerate((lo_r, hi_r)):
            with Image.open(root / rec["path"]) as im:
                axes[row, k].imshow(ImageOps.fit(im.convert("RGB"), (240, 240)))
            axes[row, k].set_title(f"{rec['split']}: {rec['class']}", fontsize=9)
            axes[row, k].axis("off")
    plt.suptitle("Same picture in two splits (top copy is removed)")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_leakage_examples.png", dpi=130)
    plt.close()

print("\n" + md)
print(f"Wrote manifests to {data_dir}, plots and reports to {out_dir}  ({summary['seconds']} s)")
