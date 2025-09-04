from pathlib import Path
import json
import random
import argparse


def _load_rows(manifest_path: Path) -> list[dict]:
    """Loads the manifest JSON file."""
    return [json.loads(l) for l in manifest_path.read_text().splitlines()]


def _by_label(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Return (neg_keys, pos_keys) using the series 'id' and label."""
    neg = []
    pos = []

    for row in rows:
        uid = row["id"]
        lbl = row["label"]
        (pos if lbl == 1 else neg).append(uid)
    return sorted(pos), sorted(neg)


def _take_split(
    keys: list[str], n_take: int, rng: random.Random
) -> tuple[set, list[str]]:
    """Performs a single random split of a list of keys."""
    rng.shuffle(keys)
    return set(keys[:n_take]), keys[n_take:]


def _stratified_key_split(
    neg_keys: list[str],
    pos_keys: list[str],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[set, set, set]:
    """
    Takes the lists of positive and negative keays and the desired ratios and returns
    three sets of keys for each split.
    """
    n_neg, n_pos = len(neg_keys), len(pos_keys)
    n_test_neg, n_val_neg = int(n_neg * test_ratio), int(n_neg * val_ratio)
    n_test_pos, n_val_pos = int(n_pos * test_ratio), int(n_pos * val_ratio)

    test_neg, neg_rem = _take_split(neg_keys[:], n_test_neg, rng)
    test_pos, pos_rem = _take_split(pos_keys[:], n_test_pos, rng)

    val_neg, neg_rem2 = _take_split(neg_rem, n_val_neg, rng)
    val_pos, pos_rem2 = _take_split(pos_rem, n_val_pos, rng)

    test_keys = test_neg | test_pos
    val_keys = val_neg | val_pos
    train_keys = set(neg_rem2) | set(pos_rem2)
    return train_keys, val_keys, test_keys


def _assign_rows(
    rows: list[dict], train_k: set, val_k: set, test_k: set
) -> dict[str, list[dict]]:
    """Assigns the full data dictionaries to their respective splits."""
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        uid = r["id"]
        if uid in test_k:
            splits["test"].append(r)
        elif uid in val_k:
            splits["val"].append(r)
        else:
            splits["train"].append(r)
    return splits


def _write_jsonl(items: list[dict], path: Path) -> None:
    """Writes the split data back to disk in JSONL format."""
    with path.open("w") as f:
        for row in items:
            f.write(json.dumps(row) + "\n")


def _count_by_label(rows: list[dict]) -> dict[int, int]:
    """Provides a summary of the splits."""
    c = {0: 0, 1: 0}
    for row in rows:
        c[int(row["label"])] += 1
    return c


def split_manifest(
    manifest_path: str,
    out_dir: str,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """
    Split manifes JSONL file into train/val/test sets for classification.

    The split is stratified by 'label' field, so the positive/negative ratio
    is preserved across splits.

    Args:
        manifest_path (str): Path to the manifest JSONL file.
        out_dir (str): Path where the split files will be written.
        val (float): Fraction of samples to include in the validation set (defaut 0.15).
        test (float): Fraction of samples to include in the test set (defaut 0.15).
        seed (int): Random seed for reproducibility.
    
    Returns:
        A dictionary with the number of samples in each split.    
    """
    rng = random.Random(seed)
    mnp = Path(manifest_path)
    rows = _load_rows(mnp)

    neg_keys, pos_keys = _by_label(rows)

    train_k, val_k, test_k = _stratified_key_split(neg_keys, pos_keys, val, test, rng)
    splits = _assign_rows(rows, train_k, val_k, test_k)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    _write_jsonl(splits["train"], outp / "train.jsonl")
    _write_jsonl(splits["val"], outp / "val.jsonl")
    _write_jsonl(splits["test"], outp / "test.jsonl")

    total = 0
    for name in ("train", "val", "test"):
        cnt = len(splits[name])
        total += cnt
        lab = _count_by_label(splits[name])
        print(f"{name}:{cnt}  (neg={lab[0]}, pos={lab[1]})")
    print(f"total: {total}")

    return {k: len(v) for k, v in splits.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    p.add_argument(
        "--out", required=True, help="Output directory for train/val/test jsonl"
    )
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    split_manifest(
        manifest_path=args.manifest,
        out_dir=args.out,
        val=args.val,
        test=args.test,
        seed=args.seed,
    )