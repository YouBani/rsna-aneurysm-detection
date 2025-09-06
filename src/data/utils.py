from pathlib import Path
from typing import Optional

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixel_data_handlers.util import apply_voi_lut

# NOTE: For RLE/JPEG transfer syntaxes, install the pixel decoders:
#   uv pip install pydicom pylibjpeg pylibjpeg-rle pylibjpeg-libjpeg
#
# All loaders below return float32 volumes in [0, 1] with shape (Z, H, W).

FIXED_CT_WINDOW: tuple[float, float] = (300.0, 1000.0)


def list_dicom_files(series_path: Path) -> list[Path]:
    """Return all *.dcm files in a series folder."""
    files = sorted(series_path.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"Path does not exist: {series_path}")
    return files


def load_slices(files: list[Path]) -> list[Dataset]:
    """
    Load DICOMs and sort by geometric Z (ImagePositionPatient[2]),
    falling back to InstanceNumber, then filename order.
    """
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


def load_series_ct(
    series_dir: str,
    target_slices: Optional[int] = None,
    window: tuple[float, float] = FIXED_CT_WINDOW,
) -> np.ndarray:
    """
    Load a CT/CTA series as (Z,H,W) float32 in [0,1].
    Always applies the fixed angiography window (center=300, width=1000) unless overridden.
    """
    files = list_dicom_files(Path(series_dir))
    ds_list = load_slices(files)

    wc, ww = window
    vol_hu = np.stack([to_hu(ds, ds.pixel_array) for ds in ds_list], axis=0)
    vol = window_hu(vol_hu, wc, ww)
    return center_pad_or_crop_z(vol, target_slices)


def load_series_mr(
    series_dir: str,
    subtype: str,
    target_slices: Optional[int] = None,
) -> np.ndarray:
    """
    Load an MR/MRA series as (Z, H, W) float32 in [0, 1].
    - MRA: VOI/LUT or per-slice min-max (to preserve vessel contrast)
    - MRI T2 / MRI T1post / MR: volume-wise z-score -> clamp -> [0, 1]
    Handles both multi-frame Enhanced MR and classic one-file-per-slice MR.
    """
    files = list_dicom_files(Path(series_dir))

    # Fast header-only peek to decide the path
    first_hdr = dcmread(files[0], stop_before_pixels=True)
    if is_multiframe(first_hdr):
        first = dcmread(str(files[0]))
        vol = first.pixel_array.astype(np.float32)
        # Try to sort frames by Z using Per-frame Functional Groups
        try:
            pffg = first.PerFrameFunctionalGroupsSequence
            z = [
                float(fr.PlanePositionSequence[0].ImagePositionPatient[2])
                for fr in pffg
            ]
            vol = vol[np.argsort(z)]
        except Exception:
            pass

        if subtype == "MRA":
            mn = vol.min(axis=(1, 2), keepdims=True)
            mx = vol.max(axis=(1, 2), keepdims=True)
            den = np.maximum(mx - mn, 1e-6)
            vol = np.clip((vol - mn) / den, 0.0, 1.0).astype(np.float32)
        else:
            vol = zscore_to_unit(vol)

        return center_pad_or_crop_z(vol, target_slices)

    # Classic MR: load & sort all slices (with pixels), then normalize
    ds_list = load_slices(files)
    if subtype == "MRA":
        sl = [apply_voi_or_minmax(ds, ds.pixel_array) for ds in ds_list]
        vol = np.stack(sl, axis=0).astype(np.float32)
    else:
        vol = np.stack([ds.pixel_array.astype(np.float32) for ds in ds_list], axis=0)
        vol = zscore_to_unit(vol)

    return center_pad_or_crop_z(vol, target_slices)


def load_series_auto(
    series_dir: str,
    subtype: Optional[str] = None,
    target_slices: Optional[int] = None,
) -> np.ndarray:
    if subtype in {"CT", "CTA"}:
        return load_series_ct(
            series_dir, target_slices=target_slices, window=FIXED_CT_WINDOW
        )
    return load_series_mr(series_dir, subtype=subtype, target_slices=target_slices)