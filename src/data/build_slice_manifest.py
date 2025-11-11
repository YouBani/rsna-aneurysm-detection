import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
from typing import Optional, Any


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def _process_one_json(json_path: Path) -> Optional[list[dict[str, Any]]]:
    """Reads one .json and returns a list of slice-level entries.
    Returns None if the .json or .npz is missing or invalid."""
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)

        series_id = meta.get("series_id")
        npz_path = series_id.with_suffix(".npz")

        if not npz_path.exists():
            logging.warning(f"Missing .npz file for {json_path.name}")
            return None

        num_slices = meta.get("shape_cropped", [0])[0]
        if num_slices <= 0:
            logging.warning(f"Invalid shape in {json_path.name}")
            return None

        patient_label = 1 if meta.get("has_aneurysm_mask", False) else 0

        positive_slice_indices = set()
        if patient_label == 1:
            try:
                with np.load(npz_path, mmap_mode="r") as data:
                    slice_sums = data["mask"].sum(axis=(1, 2))
                    positive_slice_indices = set(np.where(slice_sums > 0)[0])
            except Exception as e:
                logging.warning(f"Could not load mask for {series_id}: {e}")

        slice_entries = []
        for slice_idx in range(num_slices):
            has_aneurysm = slice_idx in positive_slice_indices
            slice_entries.append(
                {
                    "series_id": series_id,
                    "npz_path": npz_path,
                    "slice_idx": slice_idx,
                    "patient_label": patient_label,
                    "slice_has_aneurysm": has_aneurysm,
                }
            )
        return slice_entries

    except Exception as e:
        logging.error(f"Failed to process {json_path.name}: {e}")
        return None


def main():
    p = argparse.ArgumentParser(
        description="Build a slice-level manifest from processed .json metadata."
    )
    p.add_argument(
        "--preprocessed_dir",
        type=Path,
        required=True,
        help="Path to the preprocessed directory.",
    )
    p.add_argument(
        "--out_file",
        type=Path,
        required=True,
        help="Path to save the final manifest.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes to use.",
    )
    args = p.parse_args()

    logging.info(f"Scanning for .json files in :{args.preprocessed_dir}")
    json_files = sorted(list(args.preprocessed_dir.glob("*.json")))
    if not json_files:
        logging.error(f"No .json metadata files found in {args.preprocessed_dir}")
        return

    logging.info(f"Found {len(json_files)} .json files.")

    all_slice_entries = []

    with (
        Pool(processes=args.num_workers) as pool,
        tqdm(total=len(json_files), desc="Building Patient Manifest") as pbar,
    ):
        for result in pool.imap_unordered(_process_one_json, json_files):
            if result:
                all_slice_entries.extend(result)
            pbar.update(1)

    if not all_slice_entries:
        logging.error("No slice entries were generated. Exiting.")
        return

    logging.info(f"Total slices found: {len(all_slice_entries)}")

    logging.info(f"Saving manifest to {args.out_file}.")
    with open(args.out_file, "w") as f:
        for entry in all_slice_entries:
            f.write(json.dumps(entry) + "\n")

    pos_slices = sum(1 for e in all_slice_entries if e["slice_has_aneurysm"])
    logging.info(
        f"Done. Found {pos_slices} positive aneurysm slices out of {len(all_slice_entries)}."
    )


if __name__ == "__main__":
    main()
