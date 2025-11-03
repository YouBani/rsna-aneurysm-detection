import argparse
import json
import logging
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any, Optional
import numpy as np
from tqdm import tqdm
from src.data.utils import crop_and_normalize_series, resample_z


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _compute_target_z_from_mm(z_pos: np.ndarray, resample_mm: float) -> Optional[int]:
    """
    Given physical z positions (mm) and desired spacing (mm), compute the new Z.
    Returns None to indicate 'no resample'.
    """
    if z_pos is None or len(z_pos) <= 1:
        return None
    if resample_mm is None or resample_mm <= 0:
        return None

    span = float(z_pos[-1] - z_pos[0])
    if span <= 0.0 or resample_mm <= 0.0:
        return None

    target_z = int(round(span / max(resample_mm, 1e-6))) + 1
    target_z = max(target_z, 2)
    if target_z == len(z_pos):
        return None
    return target_z


def _process_one(
    args: tuple[dict[str, Any], Path, Optional[float], Optional[float]],
) -> Optional[tuple[str, str]]:
    """
    Worker: load -> crop/normalize -> (optional) resample to uniform mm -> save .npz + .json.
    Returns None on success or (series_id, error_message) on failure.
    """
    row, out_dir, resample_mm_ct, resample_mm_mr = args

    raw_id = row.get("id")
    if raw_id is None:
        return "UNKNOWN_SERIES_ID", "row['id'] is missing or None"
    series_id = str(raw_id)

    series_dir = row.get("image_path")
    if not series_dir:
        return str(series_id), "row['image_path'] missing"

    modality = row.get("subtype") or row.get("modality")
    if modality is None:
        return series_id, "row['subtype'] or row['modality'] missing"

    out_npz = Path(out_dir) / f"{series_id}.npz"
    out_json = Path(out_dir) / f"{series_id}.json"
    if out_npz.exists() and out_json.exists():
        return None

    try:
        vol, meta = crop_and_normalize_series(series_dir, modality=modality)
        bbox = meta.get("bbox", None)
        z_pos = meta.get("z_pos", None)

        is_ct = modality in {"CT", "CTA"}

        resample_mm = resample_mm_ct if is_ct else resample_mm_mr

        target_z = None
        if (
            resample_mm is not None
            and (z_pos is not None)
            and len(z_pos) == vol.shape[0]
        ):
            target_z = _compute_target_z_from_mm(
                np.asarray(z_pos, np.float32),
                float(resample_mm),
            )
            if target_z is not None:
                use_antialias = is_ct
                vol = resample_z(
                    vol,
                    target_z=target_z,
                    z_pos=np.asarray(z_pos, np.float32),
                    antialiasing=use_antialias,
                )
                if len(z_pos) > 1:
                    z_pos = np.linspace(z_pos[0], z_pos[-1], target_z, dtype=np.float32)
                else:
                    z_pos = None

        save_dict: dict[str, Any] = {
            "vol": vol.astype(np.float16),
        }
        if bbox is not None:
            save_dict["bbox"] = np.asarray(bbox, dtype=np.int32)
        if z_pos is not None:
            save_dict["z_pos"] = np.asarray(z_pos, dtype=np.float32)

        np.savez_compressed(out_npz, **save_dict)

        series_info = {
            "series_id": series_id,
            "modality": modality,
            "shape_cropped": [int(x) for x in vol.shape],
            "bbox": [int(b) for b in bbox] if bbox is not None else None,
            "has_z_pos": bool(z_pos is not None),
            "resampled": bool(target_z is not None),
            "resample_mm": None if resample_mm is None else float(resample_mm),
        }
        out_json.write_text(json.dumps(series_info, indent=2))

        return None

    except FileNotFoundError as e:
        return series_id, f"File not found: {e}"
    except Exception as e:
        logging.error(f"Error processing {series_id}")
        return series_id, f"Unexpected error: {e}"


def _read_jsonl(path: Path) -> list[dict]:
    text = path.read_text(errors="ignore")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess DICOM series to .npy files.")
    p.add_argument(
        "--jsonl_path",
        type=Path,
        required=True,
        help="Path to the train or val jsonl file.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to save the processed .npy files.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes to use.",
    )
    p.add_argument(
        "--resample-mm-ct",
        type=float,
        default=1.0,
        help="Target Z spacing for CT/CTA. Default: 1.0. Set to 0 or negative to disable.",
    )
    p.add_argument(
        "--resample-mm-mr",
        type=float,
        default=None,
        help="Target Z spacing for MR/MRA. Default: None (keep native)",
    )

    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting preprocessing for: {args.jsonl_path.name}")

    rows = _read_jsonl(args.jsonl_path)

    resample_ct = (
        args.resample_mm_ct if args.resample_mm_ct and args.resample_mm_ct > 0 else None
    )
    resample_mr = (
        args.resample_mm_mr if args.resample_mm_mr and args.resample_mm_mr > 0 else None
    )

    tasks = [(row, args.out_dir, resample_ct, resample_mr) for row in rows]

    logging.info(f"Found {len(tasks)} series to process.")

    errors = []
    success_count = 0
    with (
        Pool(processes=args.num_workers) as pool,
        tqdm(total=len(tasks), desc="Processing Series") as pbar,
    ):
        try:
            for result in pool.imap_unordered(_process_one, tasks):
                if result is None:
                    success_count += 1
                else:
                    errors.append(result)
                pbar.update(1)
        except KeyboardInterrupt:
            logging.error("Interrupted. Terminating workers")
            pool.terminate()
            pool.join()
            sys.exit(1)

    logging.info(f"Done. Succes {success_count} | Failed: {len(errors)}")
    if errors:
        logging.error("\n--- Error Summary ---")
        for series_id, err in errors:
            logging.error(f"Series ID: {series_id}\n  Error: {err}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
