import json
import argparse
import random
import csv
from pathlib import Path
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_jsonl", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--pos_pad", type=int, default=2)
    p.add_argument("--neg_per_pos", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    by_series = defaultdict(list)
    with args.manifest_jsonl.open() as f:
        for line in f:
            e = json.loads(line)
            sid = e["series_id"]
            by_series[sid].append((int(e["slice_idx"]), bool(e["slice_has_aneurysm"])))

    rows = []
    for series_id, series_slices in by_series.items():
        series_slices.sort(key=lambda x: x[0])
        num_slices = series_slices[-1][0] + 1
        pos = {z for z, has in series_slices if has}
        pos_expanded = set()
        for z in pos:
            for dz in range(-args.pos_pad, args.pos_pad + 1):
                pos_expanded.add(min(max(z + dz, 0), num_slices - 1))

        all_idx = set(range(num_slices))
        neg_pool = sorted(all_idx - pos_expanded)
        if args.neg_per_pos < 0:
            neg = set(neg_pool)
        elif len(pos_expanded) == 0 and args.neg_per_pos >= 0:
            n_neg = min(64, len(neg_pool))
            neg = set(random.sample(neg_pool, n_neg))
        else:
            n_neg = min(len(pos_expanded) * args.neg_per_pos, len(neg_pool))
            neg = set(random.sample(neg_pool, n_neg)) if n_neg > 0 else set()

        for z in sorted(pos_expanded):
            rows.append((series_id, z, 1))
        for z in sorted(neg):
            rows.append((series_id, z, 0))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_id", "slice_idx", "label"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
