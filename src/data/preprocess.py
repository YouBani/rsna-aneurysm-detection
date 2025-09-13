import argparse
import json
import numpy as np
from pathlib import Path
from src.data.utils import load_series_auto
from multiprocessing import Pool, cpu_count


def process_series(args_tuple):
    """Worker function to process a single DICOM series"""
    row, cache_dir, target_slices = args_tuple
    series_id = row["id"]
    subtype = row.get("subtype")
    series_dir = Path(row.get("image_path"))

    try:
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
    except Exception as e:
        return f"Error processing series {series_id}: {e}"


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess DICOM series to .npy files.")
    p.add_argument(
        "--jsonl_path", type=Path, help="Path to the train or val jsonl file."
    )
    p.add_argument(
        "--cache_dir", type=Path, help="Directory to save the processed .npy files."
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
    """
    Main function to orchestrate the offline preprocessing.
    """
    args = parse_args()
    print(f"Starting preprocessing for: {args.jsonl_path.name}")

    with open(args.jsonl_path) as f:
        rows = [json.loads(line) for line in f]

    tasks = [(row, args.cache_dir, args.target_slices) for row in rows]

    # Use a multiplrocessing Pool to process the data in parallel
    print(f"Found {len(tasks)} series to process.")
    errors = []
    with Pool(processes=args.num_workers) as pool:
        for result in pool.imap_unordered(process_series, tasks):
            if result is not None:
                errors.append(result)

    print("\nPreprocessing finished.")

    if errors:
        print(f"\nEncountered {len(errors)} errors during processing:")
        for err in errors:
            print(err)
    else:
        print("All series processed successfully.")


if __name__ == "__main__":
    main()
# ```

### How to Use It

# 1.  **Save the file** as `src/preprocess.py`.
# 2.  **Run it from your terminal** for both your training and validation sets. You'll need to adjust the paths to match your system.

# **Example for your training data:**
# ```bash
# python -m src.preprocess \
#     /fsx/path/to/your/train.jsonl \
#     /fsx/path/to/raw/dicom/data \
#     /fsx/cache_z64 \
#     --target_slices 64 \
#     --num_workers 4
# ```

# **Example for your validation data:**
# ```bash
# python -m src.preprocess \
#     /fsx/path/to/your/val.jsonl \
#     /fsx/path/to/raw/dicom/data \
#     /fsx/cache_z64 \
#     --target_slices 64 \
#     --num_workers 8
