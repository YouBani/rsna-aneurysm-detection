

from pathlib import Path
from typing import Optional

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixel_handlers.util import apply_voi_lut


def list_dicom_files(series_path: Path) -> list[Path]:
    """Return all *.dcm files in a series folder."""
    files = sorted(series_path.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"Path does not exist: {series_path}")
    return files


def load_slices(files: list[Path]) -> list[Dataset]:
    """Load DICOMs and sort by InstanceNumber, fallback to filename."""
    ds_list = [dcmread(str(fp)) for fp in files]
    try:
        ds_list.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    except Exception:
        # Some series may have missing InstanceNumber; fallback to filename order
        try:
            ds_list.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
        except Exception:
            ds_list = [dcmread(str(fp)) for fp in sorted(files)]
    return ds_list


def is_multiframe(ds: Dataset) -> bool:
    """Enhanced MR often stores an entire stack as one multi-frame DICOM."""
    return hasattr(ds, "NumberOfFrames")


def to_hu(ds, arr: np.ndarray) -> np.ndarray:
    """CT: convert raw pixels to Hounsfield Units using slope/intercept."""
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr.astype(np.float32) * slope + intercept


def window_hu(vol: np.ndarray, center: float, width: float) -> np.ndarray:
    """Perform windowing operation by Cliping HU to [C - W/2, C + W/2] and scaling to [0, 1]."""
    lo, hi = center - width / 2.0, center + width / 2.0
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / max(hi - lo, 1e-6)
    return vol.astype(np.float32)


def apply_voi_or_minmax(ds, arr: np.ndarray) -> np.ndarray:
    """MR/MRA: apply VOI LUT, if it fails fall back to simple min-max normalization"""
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
        a_min, a_max = float(arr.min()), float(arr.max())
        return (
            ((arr - a_min) / (a_max - a_min))
            if a_max > a_min
            else np.zeros_like(arr, dtype=np.float32)
        )
    except Exception:
        arr = arr.astype(np.float32)
        a_min, a_max = float(arr.min()), float(arr.max())
        return (
            ((arr - a_min) / (a_max - a_min))
            if a_max > a_min
            else np.zeros_like(arr, dtype=np.float32)
        )


def zscore_to_unit(vol: np.ndarray, clamp: float = 5.0) -> np.ndarray:
    """MR (non-angio): volume-wise z-score, clamp to ±clamp, map to [0, 1]."""
    m, s = float(vol.mean()), float(vol.std() + 1e-6)
    vol = (vol - m) / s
    vol = np.clip(vol, -clamp, clamp)
    vol = (vol + clamp) / (2 * clamp)
    return vol.astype(np.float32)


def center_pad_or_crop_z(vol: np.ndarray, target_z: Optional[int]) -> np.ndarray:
    """Center pad or crop a 3D volume along the Z-axis to a target size."""
    if target_z is None:
        return vol
    z, h, w = vol.shape
    if z == target_z:
        return vol
    if z < target_z:
        pad = target_z - z
        p0, p1 = pad // 2, pad - pad // 2
        return np.pad(vol, ((p0, p1), (0, 0), (0, 0)), mode="constant")
    start = (z - target_z) // 2
    return vol[start : start + target_z]
