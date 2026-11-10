"""Create a small synthetic stand-in for AIDERv2 with the same folder layout.

It is for smoke-testing the pipeline and developing the app without the 1.5 GB download.
The images are generated textures, NOT real aerial photos. It also plants known problems
(exact duplicates, a train-to-test leak, a corrupt file, a label conflict) so the duplicate
checker can be tested against a known answer.

    python scripts/make_demo_data.py --out demo_data
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronerescue import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="demo_data")
ap.add_argument("--train", type=int, default=40, help="images per class in Train")
ap.add_argument("--val", type=int, default=10)
ap.add_argument("--test", type=int, default=10)
ap.add_argument("--zips", action="store_true", help="also write Train.zip, Val.zip, Test.zip under <out>/zips")
args = ap.parse_args()

rng = np.random.default_rng(7)
BASE = {"Earthquake": (130, 120, 110), "Flood": (60, 110, 170),
        "Normal": (70, 140, 80), "Wildfire": (200, 90, 40)}


def make_image(cls):
    h, w = int(rng.integers(260, 700)), int(rng.integers(260, 700))
    base = np.array(BASE[cls], dtype=float)
    img = np.ones((h, w, 3)) * base + rng.normal(0, 18, (h, w, 3))
    for _ in range(int(rng.integers(6, 14))):          # random blobs make every image unique
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        r = int(rng.integers(15, 80))
        colour = base + rng.normal(0, 60, 3)
        yy, xx = np.ogrid[:h, :w]
        img[(yy - y) ** 2 + (xx - x) ** 2 < r * r] = colour
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.2))


root = Path(args.out) / C.DATASET_DIRNAME
if root.exists():
    shutil.rmtree(root)
counts = {"Train": args.train, "Val": args.val, "Test": args.test}
for split in C.SPLITS:
    for cls in C.CLASSES:
        d = root / split / cls
        d.mkdir(parents=True)
        for i in range(1, counts[split] + 1):
            make_image(cls).save(d / f"img ({i}).jpg", quality=90)

# ---- planted problems (answer key printed at the end) ----
train_flood = sorted((root / "Train/Flood").glob("*.jpg"))
train_fire = sorted((root / "Train/Wildfire").glob("*.jpg"))
for k in range(3):                                      # 3 exact copies inside Train
    shutil.copy(train_flood[k], root / "Train/Flood" / f"copy_{k}.jpg")
for k in range(4):                                      # 4 Train images leaked into Test, re-saved smaller
    im = Image.open(train_fire[k + 5])
    im.resize((int(im.width * 0.8), int(im.height * 0.8))).save(root / "Test/Wildfire" / f"leak_{k}.jpg", quality=70)
val_norm = sorted((root / "Val/Normal").glob("*.jpg"))
for k in range(2):                                      # 2 Val images also present in Test
    shutil.copy(val_norm[k], root / "Test/Normal" / f"valdup_{k}.jpg")
shutil.copy(train_flood[10], root / "Train/Normal" / "conflict_0.jpg")   # same picture, two labels
(root / "Train/Earthquake" / "corrupt_0.jpg").write_bytes(b"\xff\xd8\xff\xe0 not a real jpeg")

print("Planted problems (answer key):")
print("  3 exact duplicates inside Train/Flood")
print("  4 near-duplicates: Train/Wildfire images copied into Test/Wildfire (resized, re-compressed)")
print("  2 duplicates: Val/Normal images copied into Test/Normal")
print("  1 label conflict: a Train/Flood image also in Train/Normal")
print("  1 corrupt file in Train/Earthquake")

if args.zips:
    zdir = Path(args.out) / "zips"
    zdir.mkdir(exist_ok=True)
    for split in C.SPLITS:
        with zipfile.ZipFile(zdir / f"{split}.zip", "w", zipfile.ZIP_STORED) as z:
            for p in (root / split).rglob("*"):
                if p.is_file():     # wrapper folder inside the zip, like a real archive might have
                    z.write(p, Path("data/Dataset_Images") / p.relative_to(root))
    print(f"Wrote zips to {zdir}")
