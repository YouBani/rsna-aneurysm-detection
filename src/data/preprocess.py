import argparse
import json
import logging
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from src.data.utils import load_series_auto


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _process_series(
    args_tuple: tuple[dict[str, Any], Path, int],
) -> Optional[tuple[str, str]]:
    """
    Worker function to process a single DICOM series.
    Returns None on success or an (series_id, error_message) on failure.
    """
    row, cache_dir, target_slices = args_tuple
    series_id = row.get("id")
    subtype = row.get("subtype")
    series_dir_str = row.get("image_path")

    if not series_dir_str:
        return str(series_id), "image_path key is missing or empty in jslon row."

    series_dir = Path(series_dir_str)

    try:
        logging.info(f"Worker starting on series: {series_id}")
        output_path = cache_dir / f"{series_id}.npy"
        if output_path.exists():
            return None

        vol = load_series_auto(
            series_dir=str(series_dir),
            target_slices=target_slices,
            subtype=subtype,
        )

        np.save(output_path, vol)
        return None
    except FileNotFoundError as e:
        return str(series_id), f"File not found error: {e}"
    except Exception as e:
        return str(series_id), f"An unexpected error occured: {e}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess DICOM series to .npy files.")
    p.add_argument(
        "--jsonl_path",
        type=Path,
        required=True,
        help="Path to the train or val jsonl file.",
    )
    p.add_argument(
        "--cache_dir",
        type=Path,
        required=True,
        help="Directory to save the processed .npy files.",
    )
    p.add_argument(
        "--target_slices",
        type=int,
        default=64,
        help="Target number of slices after resampling.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes to use.",
    )
    return p.parse_args()


def main():
    """Main function to orchestrate the offline preprocessing."""
    args = parse_args()

    logging.info(f"Starting preprocessing for: {args.jsonl_path.name}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output will be saved to: {args.cache_dir}")

    with open(args.jsonl_path) as f:
        rows = [json.loads(line) for line in f]

    tasks: list[tuple] = [(row, args.cache_dir, args.target_slices) for row in rows]
    logging.info(f"Found {len(tasks)} series to process.")

    errors = []
    success_count = 0

    pool = Pool(processes=args.num_workers)
    try:
        with tqdm(total=len(tasks), desc="Processing Series") as pbar:
            for result in pool.imap_unordered(_process_series, tasks):
                if result is not None:
                    series_id, error_msg = result
                    errors.append((series_id, error_msg))
                    logging.warning(f"Failed series {series_id}: {error_msg}")
                else:
                    success_count += 1
                pbar.update(1)

    finally:
        logging.info("All tasks processed. Shutting down worker pool...")
        pool.close()
        pool.join(timeout=10)
        pool.terminate()
        logging.info("Worker pool has been shut down.")

    logging.info("\n--- Preprocessing Finished ---")
    logging.info(f"Successfully processed: {success_count} series")
    logging.info(f"Failed to process: {len(errors)} series")

    if errors:
        logging.error("\n--- Error Summary ---")
        for series_id, err in errors:
            logging.error(f"Series ID: {series_id}\n  Error: {err}\n")
        sys.exit(1)
    else:
        logging.info("All series processed successfully.")


if __name__ == "__main__":
    main()
