"""Duplicate-detection helpers shared by the Phase 0 scripts."""
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import config as C


def analyze_image(root, rel):
    """Read one image: size, mean colour, perceptual hash and a 16x16 grey thumbnail."""
    path = Path(root) / rel
    res = {"path": rel, "ok": False, "error": ""}
    try:
        with Image.open(path) as im:
            im.load()                               # full decode: catches truncated files
            w, h = im.size
            rgb = im.convert("RGB")
        small = rgb.resize((16, 16), Image.LANCZOS)
        thumb = np.asarray(small.convert("L"), dtype=np.uint8)
        g = np.asarray(rgb.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).flatten()      # difference hash (dHash): 64 bits
        mean = np.asarray(small, dtype=float).reshape(-1, 3).mean(axis=0)
        res.update(ok=True, width=w, height=h, size_bytes=path.stat().st_size,
                   dhash=int.from_bytes(np.packbits(bits).tobytes(), "big"),
                   mean_r=mean[0], mean_g=mean[1], mean_b=mean[2], thumb=thumb.tobytes())
    except Exception as e:                          # noqa: BLE001 - any decode problem means "corrupt"
        res["error"] = f"{type(e).__name__}: {e}"[:120]
    return res


def candidate_pairs(hashes, max_dist=C.HASH_MAX_DISTANCE, bucket_cap=400):
    """All pairs whose 64-bit hashes differ in at most `max_dist` bits, as {(i, j): distance}.

    Multi-index hashing: two hashes within `max_dist` bits must share one of `max_dist + 1`
    slices exactly, so only images that land in the same bucket are compared."""
    n_chunks = max_dist + 1
    edges = np.linspace(0, 64, n_chunks + 1).astype(int)
    found, skipped = {}, 0
    for c in range(n_chunks):
        lo, hi = int(edges[c]), int(edges[c + 1])
        mask = (1 << (hi - lo)) - 1
        buckets = defaultdict(list)
        for i, hsh in enumerate(hashes):
            buckets[(hsh >> lo) & mask].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > bucket_cap:           # e.g. a pile of near-blank images
                skipped += 1
                members = members[:bucket_cap]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    d = (hashes[i] ^ hashes[j]).bit_count()
                    if d <= max_dist:
                        found[(i, j)] = d
    return found, skipped


def pair_mad(thumbs, pairs):
    """Mean absolute difference (0..255) between the 16x16 grey thumbnails of each pair."""
    return np.array([np.abs(thumbs[i].astype(np.int16) - thumbs[j].astype(np.int16)).mean()
                     for i, j in pairs])


def union_find_groups(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in pairs:
        parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def reset_duplicate_columns(df):
    df["dup_group"] = -1
    df["label_conflict"] = False
    df["test_leak_exposed"] = False
    df.loc[df["status"].isin(["dup_leak", "dup_leak_test", "dup_within", "dup_conflict"]), "status"] = "keep"


def assign_status(df, groups, v_idx):
    """Apply the cleaning rules to duplicate groups, modifying df in place. Returns counts.

    Test is never changed. Val loses copies of Test images, Train loses copies of Val/Test images,
    copies inside one split are reduced to one, and the same picture under different labels is
    dropped from Train/Val (which label is right is unknown)."""
    reset_duplicate_columns(df)
    stats = {"leak_train_in_test": 0, "leak_train_in_val": 0, "leak_val_in_test": 0,
             "within_test_copies": 0, "test_exposed": 0}
    for gid, members in enumerate(groups):
        idx = [v_idx[m] for m in members]
        g = df.loc[idx]
        by = {s: sorted(g[g["split"] == s].index, key=lambda i: df.at[i, "path"]) for s in C.SPLITS}
        df.loc[idx, "dup_group"] = gid
        conflict = g["class"].nunique() > 1
        if conflict:
            df.loc[idx, "label_conflict"] = True
        stats["within_test_copies"] += max(len(by["Test"]) - 1, 0)
        if by["Test"] and (by["Train"] or by["Val"]):
            df.loc[by["Test"], "test_leak_exposed"] = True      # this test image has a look-alike in training data
            stats["test_exposed"] += len(by["Test"])
        if by["Val"]:
            if by["Test"]:
                df.loc[by["Val"], "status"] = "dup_leak_test"
                stats["leak_val_in_test"] += len(by["Val"])
            elif conflict:
                df.loc[by["Val"], "status"] = "dup_conflict"
            else:
                df.loc[by["Val"][1:], "status"] = "dup_within"
        if by["Train"]:
            if by["Test"] or by["Val"]:
                df.loc[by["Train"], "status"] = "dup_leak"
                stats["leak_train_in_test"] += len(by["Train"]) if by["Test"] else 0
                stats["leak_train_in_val"] += len(by["Train"]) if by["Val"] else 0
            elif conflict:
                df.loc[by["Train"], "status"] = "dup_conflict"
            else:
                df.loc[by["Train"][1:], "status"] = "dup_within"
    return stats


def assign_status_pairwise(df, pairs, dists, v_idx):
    """Pair-level cleaning (the default). A Train/Val image is removed only if it DIRECTLY resembles an
    image in a higher-priority split, so a chain of similar video frames is not wiped out because one of
    its frames landed in Test.

    Rules (priority Test > Val > Train; Test is never changed):
      - Val image resembling a Test image         -> dropped  (dup_leak_test)
      - Train image resembling a Val/Test image   -> dropped  (dup_leak)
      - exact copy (hash distance 0) inside Train or Val -> keep the first, drop the rest (dup_within)
      - the pair has two different class labels   -> Train/Val member dropped (dup_conflict)
      - merely similar images inside one split are kept (redundant, but not leakage)
    Test images with a look-alike in Train/Val are flagged test_leak_exposed."""
    reset_duplicate_columns(df)
    rank = {"Train": 0, "Val": 1, "Test": 2}
    within, conflict, leak_train, leak_val, exposed, test_internal = set(), set(), set(), set(), set(), set()
    leak_train_test, leak_train_val = set(), set()
    for (i, j), d in zip(pairs, dists):
        a, b = v_idx[i], v_idx[j]
        sa, sb = df.at[a, "split"], df.at[b, "split"]
        if df.at[a, "class"] != df.at[b, "class"]:
            conflict.update((a, b))
        if sa == sb:
            if sa == "Test":
                test_internal.update((a, b))
            elif d == 0:
                within.add(b if df.at[a, "path"] < df.at[b, "path"] else a)
            continue
        hi, lo = (a, b) if rank[sa] > rank[sb] else (b, a)
        shi, slo = df.at[hi, "split"], df.at[lo, "split"]
        if shi == "Test":
            exposed.add(hi)
        if slo == "Train":
            leak_train.add(lo)
            (leak_train_test if shi == "Test" else leak_train_val).add(lo)
        else:
            leak_val.add(lo)
    for idx in within:
        df.at[idx, "status"] = "dup_within"
    for idx in conflict:
        if df.at[idx, "split"] != "Test":
            df.at[idx, "status"] = "dup_conflict"
    for idx in leak_train:
        df.at[idx, "status"] = "dup_leak"
    for idx in leak_val:
        df.at[idx, "status"] = "dup_leak_test"
    for idx in conflict:
        df.at[idx, "label_conflict"] = True
    for idx in exposed:
        df.at[idx, "test_leak_exposed"] = True
    for gid, members in enumerate(union_find_groups(len(v_idx), pairs)):
        df.loc[[v_idx[m] for m in members], "dup_group"] = gid
    return {"leak_train_in_test": len(leak_train_test), "leak_train_in_val": len(leak_train_val),
            "leak_val_in_test": len(leak_val), "within_test_copies": len(test_internal),
            "test_exposed": len(exposed)}
