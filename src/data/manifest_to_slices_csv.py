import json
import argparse
import random
import csv
import logging
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest_jsonl",
        type=Path,
        required=True,
        help="Path to the FULL slice manifest",
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="Path to save the final sampled .csv manifest",
    )
    p.add_argument(
        "--pos_pad",
        type=int,
        default=2,
        help="Number of slices to pad around a positive slice",
    )
    p.add_argument(
        "--neg_per_pos",
        type=int,
        default=3,
        help="Number of negative slices to sample for each positive slice.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = p.parse_args()
    random.seed(args.seed)

    logging.info(f"Loading full slice manifest from: {args.manifest_jsonl}")

    with args.manifest_jsonl.open() as f:
        full_manifest: list[dict[str, Any]] = [
            json.loads(line) for line in tqdm(f, desc="Loading manifest")
        ]

    by_series = defaultdict(list)
    for entry in full_manifest:
        by_series[entry["series_id"]].append(entry)

    final_rows_to_write = []

    for series_id, series_slices in by_series.items():
        series_slices.sort(key=lambda x: x[0])
        num_slices = series_slices[-1][0] + 1
        pos = {z for z, has in series_slices if has}
        pos_expanded_idx = set()
        for z in pos:
            for dz in range(-args.pos_pad, args.pos_pad + 1):
                pos_expanded_idx.add(min(max(z + dz, 0), num_slices - 1))

        all_slice_idx = set(range(num_slices))
        neg_pool_idx = sorted(all_slice_idx - pos_expanded_idx)
        neg_sampled_idx = set()

        if args.neg_per_pos < 0:
            neg_sampled_idx = set(neg_pool_idx)
        elif len(pos_expanded_idx) == 0 and args.neg_per_pos >= 0:
            n_neg = min(64, len(neg_pool_idx))
            neg_sampled_idx = set(random.sample(neg_pool_idx, n_neg))
        else:
            n_neg = min(len(pos_expanded_idx) * args.neg_per_pos, len(neg_pool_idx))
            neg_sampled_idx = (
                set(random.sample(neg_pool_idx, n_neg)) if n_neg > 0 else set()
            )

        slice_map = {s["slice_idx"]: s for s in series_slices}

        for z in sorted(pos_expanded_idx):
            entry = slice_map[z]
            final_rows_to_write.append(
                (
                    entry["series_id"],
                    entry["vol_path"],
                    entry["mask_path"],
                    entry["slice_idx"],
                    1,
                )
            )

        for z in sorted(pos_expanded_idx):
            entry = slice_map[z]
            final_rows_to_write.append(
                (
                    entry["series_id"],
                    entry["vol_path"],
                    entry["mask_path"],
                    entry["slice_idx"],
                    0,
                )
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing {len(final_rows_to_write)} sampled slices to {args.out_csv}")

    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series_id", "vol_path", "mask_path", "slice_idx", "label"])
        w.writerows(final_rows_to_write)

    logging.info("Done.")


if __name__ == "__main__":
    main()
