"""Phase 1, step 1: resize every cleaned image once and store it as a uint8 array.

All Phase 1-3 models read these arrays, so they all see identical inputs and training is not slowed by
decoding JPEGs. Images are squashed to a square (the dataset itself uses 224x224 squares).

    python scripts/03_build_cache.py --size 128
    python scripts/03_build_cache.py --size 128 --variant original   # optional: the uncleaned official split
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronerescue import config as C
from dronerescue.pipeline import cache_dir, load_manifest_clean

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(C.DATA_DIR))
ap.add_argument("--size", type=int, default=C.DEFAULT_IMG_SIZE)
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--variant", choices=["cleaned", "original"], default="cleaned",
                help="cleaned: de-duplicated split (default). original: every readable image of the official split, "
                     "used to measure how much leakage inflates scores")
args = ap.parse_args()
data_dir = Path(args.data_dir)
root = data_dir / C.DATASET_DIRNAME


def load_resize(size, rel):
    with Image.open(root / rel) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.uint8)


if args.variant == "original":
    full = pd.read_csv(data_dir / "manifest.csv")
    man = full[~full["status"].isin(["corrupt", "too_small"])][["path", "split", "class", "test_leak_exposed"]]
else:
    man = load_manifest_clean(data_dir)
if "test_leak_exposed" not in man:
    man["test_leak_exposed"] = False
out = cache_dir(data_dir, args.size, args.variant)
out.mkdir(parents=True, exist_ok=True)
for split in C.SPLITS:
    rows = man[man["split"] == split].reset_index(drop=True)
    print(f"[{split}] resizing {len(rows):,} images to {args.size}x{args.size} ...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        x = np.stack(list(ex.map(partial(load_resize, args.size), rows["path"].tolist(), chunksize=64)))
    y = rows["class"].map(C.CLASS_TO_IDX).to_numpy(dtype=np.int64)
    np.savez(out / f"{split}.npz", x=x, y=y, exposed=rows["test_leak_exposed"].to_numpy(dtype=bool),
             paths=rows["path"].to_numpy(dtype=str))
    counts = {c: int((y == i).sum()) for i, c in enumerate(C.CLASSES)}
    print(f"  saved {split}.npz  x={x.shape}  classes={counts}  exposed={int(rows['test_leak_exposed'].sum())}")
print(f"Cache written to {out}")
