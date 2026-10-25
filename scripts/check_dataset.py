"""Validate COD10K: image/mask pairing, sizes, corrupt files, mask values, and (near-)duplicate leakage across splits."""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from tqdm import tqdm

from cod.config import load_config
from cod.data.dataset import build_samples


def dhash(path, n=8) -> int:
    g = np.asarray(Image.open(path).convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--max-hamming", type=int, default=2, help="near-duplicate threshold on 64-bit dHash")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    seed = cfg["experiment"]["seed"]
    problems, hashes = 0, {}
    for split in ("train", "val", "test"):
        samples = build_samples(cfg["data"], split, seed)
        empty = 0
        for s in tqdm(samples, desc=split):
            try:
                im, mk = Image.open(s.image), Image.open(s.mask)
                im.load(), mk.load()
                m = np.asarray(mk.convert("L"))
                if im.size != mk.size:
                    problems += 1
                    print("size mismatch", s.name, im.size, mk.size)
                if im.mode not in ("RGB", "L"):
                    print("note: mode", im.mode, s.name)
                if not set(np.unique(m)).issubset({0, 255}) and len(np.unique(m)) > 2:
                    print("note: non-binary mask values", s.name)
                empty += int((m > 127).sum() == 0)
                hashes[s.name] = (split, dhash(s.image))
            except Exception as e:
                problems += 1
                print("CORRUPT", s.name, e)
        print(f"{split}: {len(samples)} samples, {empty} empty masks")
    # cross-split near-duplicates via Hamming distance on dHash
    items = list(hashes.items())
    arr = np.array([h for _, (_, h) in items], dtype=object)
    bits = np.array([[(int(h) >> i) & 1 for i in range(64)] for h in arr], dtype=np.uint8)
    leaks = defaultdict(list)
    for i in range(len(items)):
        d = (bits[i + 1:] != bits[i]).sum(1)
        for j in np.nonzero(d <= a.max_hamming)[0] + i + 1:
            (n1, (s1, _)), (n2, (s2, _)) = items[i], items[j]
            if s1 != s2:
                leaks[(s1, s2)].append((n1, n2, int(d[j - i - 1])))
    for k, v in leaks.items():
        print(f"near-duplicate pairs {k}: {len(v)}", v[:3])
    print("problems:", problems)


if __name__ == "__main__":
    main()
