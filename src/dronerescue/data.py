"""Download, unpack and index the AIDERv2 dataset."""
import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from . import config as C


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_file(url, dest, md5=None):
    """Download url to dest, skipping the download when a verified copy already exists."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (md5 is None or md5sum(dest) == md5):
        print(f"  already downloaded: {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    start, last = time.time(), 0
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if done - last >= 50 * (1 << 20):
                last = done
                pct = f" ({100 * done / total:.0f}%)" if total else ""
                print(f"    {done / 1e6:,.0f} MB{pct}  {done / 1e6 / max(time.time() - start, 1e-6):.1f} MB/s")
                sys.stdout.flush()
    tmp.rename(dest)
    if md5 is not None:
        got = md5sum(dest)
        if got != md5:
            raise RuntimeError(f"Checksum mismatch for {dest.name}: expected {md5}, got {got}. "
                               "Delete the file and download again.")
        print(f"  checksum OK: {dest.name}")
    return dest


def extract_zip(zip_path, target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target_dir)


def canonical_class(folder_name):
    """Map a folder name such as 'Fire', 'floods' or 'Normal' to one of C.CLASSES, or None."""
    key = folder_name.strip().lower().replace(" ", "_").replace("-", "_")
    return C.CLASS_ALIASES.get(key)


def describe_tree(path, max_depth=3, max_lines=40):
    """A short directory listing used in error messages so you can see what the archive really contains."""
    path = Path(path)
    lines = []
    for p in sorted(path.rglob("*")):
        depth = len(p.relative_to(path).parts)
        if p.is_dir() and depth <= max_depth:
            n_files = sum(1 for f in p.iterdir() if f.is_file())
            lines.append(f"  {'  ' * (depth - 1)}{p.name}/  ({n_files} files directly inside)")
        if len(lines) >= max_lines:
            lines.append("  ...")
            break
    return "\n".join(lines) if lines else "  (empty)"


def find_class_root(split_dir):
    """The zips may contain extra wrapper folders. Return the folder that directly holds the class folders."""
    split_dir = Path(split_dir)
    wanted = set(C.CLASSES)
    candidates = [split_dir] + [p for p in split_dir.rglob("*") if p.is_dir()]
    for cand in sorted(candidates, key=lambda p: len(p.parts)):
        have = {canonical_class(p.name) for p in cand.iterdir() if p.is_dir()} - {None}
        if wanted <= have:
            return cand
    raise FileNotFoundError(
        f"Could not find the four class folders {C.CLASSES} under {split_dir}.\n"
        f"What is actually there:\n{describe_tree(split_dir)}\n"
        "If the folder names differ, add them to CLASS_ALIASES in src/dronerescue/config.py."
    )


def list_images(data_dir=None):
    """Return a list of (relative_path, split, class_name) for every image file in the dataset."""
    data_dir = Path(data_dir or C.DATA_DIR)
    root = data_dir / C.DATASET_DIRNAME
    rows = []
    for split in C.SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"{split_dir} not found. Run scripts/00_download_data.py first.")
        class_root = find_class_root(split_dir)
        for cls_dir in sorted(p for p in class_root.iterdir() if p.is_dir()):
            cls = canonical_class(cls_dir.name)
            if cls is None:
                continue
            for p in sorted(cls_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in C.IMAGE_EXTS and not p.name.startswith("._"):
                    rows.append((p.relative_to(root).as_posix(), split, cls))
    return rows
