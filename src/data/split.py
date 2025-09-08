from pathlib import Path
import json
import random
import argparse


def _load_rows(manifest_path: Path) -> list[dict]:
    """Loads the manifest JSON file."""
    return [json.loads(l) for l in manifest_path.read_text().splitlines()]


def _write_jsonl(items: list[dict], path: Path) -> None:
    """Writes the split data back to disk in JSONL format."""
    with path.open("w") as f:
        for row in items:
            f.write(json.dumps(row) + "\n")


def _get_pid(row: dict) -> str:
    if "patient_id" in row: return str(row["patient_id"])
    raise KeyError("Row is missing patient id.")


def _patients_by_label(rows: list[dict]) -> tuple[list[str], list[str]]:
    """
    Build patient-level labels.
    Returns (pos_patient_ids, neg_patient_ids).
    """
    per_patient_pos = {}
    for row in rows:
        pid = _get_pid(row)
        lbl = int(row["label"])
        per_patient_pos[pid] = per_patient_pos.get(pid, 0) or lbl
    
    pos_pids = [pid for pid, v in per_patient_pos.items() if v == 1]
    neg_pids = [pid for pid, v in per_patient_pos.items() if v == 0]
    return sorted(pos_pids), sorted(neg_pids)


def _take_split(
    keys: list[str], n_take: int, rng: random.Random
) -> tuple[set, list[str]]:
    """Performs a single random split of a list of keys."""
    rng.shuffle(keys)
    n_take = min(n_take, len(keys))
    return set(keys[:n_take]), keys[n_take:]


def _stratified_group_split(
    neg_pids: list[str],
    pos_pids: list[str],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[set, set, set]:
    """
    Split on PATIENTS, stratified by patient label.
    """
    n_neg, n_pos = len(neg_pids), len(pos_pids)
    n_test_neg, n_val_neg = int(n_neg * test_ratio), int(n_neg * val_ratio)
    n_test_pos, n_val_pos = int(n_pos * test_ratio), int(n_pos * val_ratio)

    test_neg, neg_rem = _take_split(neg_pids[:], n_test_neg, rng)
    test_pos, pos_rem = _take_split(pos_pids[:], n_test_pos, rng)

    val_neg,  neg_rem2 = _take_split(neg_rem, n_val_neg, rng)
    val_pos,  pos_rem2 = _take_split(pos_rem, n_val_pos, rng)

    test_p = test_neg | test_pos
    val_p  = val_neg  | val_pos
    train_p = set(neg_rem2) | set(pos_rem2)
    return train_p, val_p, test_p


def _assign_rows_by_patient(
    rows: list[dict], train_p: set, val_p: set, test_p: set
) -> dict[str, list[dict]]:
    """Assign series rows to splits based on their patient id."""
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        pid = _get_pid(row)
        if pid in test_p:
            splits["test"].append(row)
        elif pid in val_p:
            splits["val"].append(row)
        else:
            splits["train"].append(row)
    return splits


def _count_series_by_label(rows: list[dict]) -> dict[int, int]:
    """Provide a summary of the splits."""
    c = {0: 0, 1: 0}
    for row in rows:
        c[int(row["label"])] += 1
    return c


def _count_patients_by_label(rows: list[dict]) -> dict[int, int]:
    """Patient-level counts from series rows."""
    pos = set()
    neg = set()
    seen = {}
    for row in rows:
        pid = _get_pid(row)
        lbl = int(row["label"])
        seen[pid] = seen.get(pid, 0) or lbl
    for pid, lbl in seen.items():
        (pos if lbl else neg).add(pid)
    return {0: len(neg), 1: len(pos)}


def _check_no_leak(train_p: set[str], val_p: set[str], test_p: set[str]) -> None:
    assert train_p.isdisjoint(val_p)
    assert train_p.isdisjoint(test_p)
    assert val_p.isdisjoint(test_p)


def split_manifest(
    manifest_path: str,
    out_dir: str,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """
    Split manifes JSONL file into train/val/test sets for classification.

    Grouped (patient-level) stratified split.

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
    rows = _load_rows(Path(manifest_path))

    neg_pids, pos_pids = _patients_by_label(rows)
    train_p, val_p, test_p = _stratified_group_split(neg_pids, pos_pids, val, test, rng)
    _check_no_leak(train_p, val_p, test_p)

    splits = _assign_rows_by_patient(rows, train_p, val_p, test_p)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    _write_jsonl(splits["train"], outp / "train.jsonl")
    _write_jsonl(splits["val"], outp / "val.jsonl")
    _write_jsonl(splits["test"], outp / "test.jsonl")

    total_series = 0
    for name in ("train", "val", "test"):
        srows = splits[name]
        total_series += len(srows)
        s_cnt = _count_series_by_label(srows)
        p_cnt = _count_patients_by_label(srows)
        print(f"{name}: series={len(srows)} (neg={s_cnt[0]}, pos={s_cnt[1]}); "
              f"patients={p_cnt[0]+p_cnt[1]} (negP={p_cnt[0]}, posP={p_cnt[1]})")
    print(f"total series: {total_series}")

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