"""Phase 0, step 1: download AIDERv2 from Zenodo, verify checksums and unpack it.

    python scripts/00_download_data.py                    # download everything (about 1.5 GB)
    python scripts/00_download_data.py --splits Val Test  # only the small splits

Files go to $DRONERESCUE_DATA (default: ./data). Re-running is safe: finished downloads are skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dronerescue import config as C
from dronerescue.data import download_file, extract_zip

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(C.DATA_DIR))
ap.add_argument("--splits", nargs="+", default=C.SPLITS, choices=C.SPLITS)
ap.add_argument("--base-url", default=C.ZENODO_BASE, help="override the download host (used for testing)")
ap.add_argument("--no-verify", action="store_true", help="skip the md5 check (testing only)")
args = ap.parse_args()

data_dir = Path(args.data_dir)
print(f"Data directory: {data_dir}")
for split in args.splits:
    name, md5 = C.ZIP_FILES[split]
    zip_path = data_dir / "raw" / name
    url = f"{args.base_url}/{name}" + ("?download=1" if args.base_url == C.ZENODO_BASE else "")
    print(f"[{split}]")
    download_file(url, zip_path, None if args.no_verify else md5)
    target = data_dir / C.DATASET_DIRNAME / split
    if target.exists() and any(target.iterdir()):
        print(f"  already extracted: {target}")
    else:
        print(f"  extracting to {target}")
        extract_zip(zip_path, target)
print("\nDone. Next: python scripts/01_eda_dedupe.py")
