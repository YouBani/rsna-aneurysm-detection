from pathlib import Path
from typing import Optional

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixel_data_handlers.util import apply_voi_lut

from scipy.ndimage import gaussian_filter1d as _gauss1d

# NOTE: For RLE/JPEG transfer syntaxes, install the pixel decoders:
#   uv pip install pydicom pylibjpeg pylibjpeg-rle pylibjpeg-libjpeg
#
# All loaders below return float32 volumes in [0, 1] with shape (Z, H, W).

FIXED_CT_WINDOW: tuple[float, float] = (300.0, 1000.0)


def safe_float(value, default: float = -1.0) -> float:
    """Safely converts a value to a float, handling None or errors."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def list_dicom_files(series_path: Path) -> list[Path]:
    """Return all *.dcm files in a series folder."""
    files = sorted(series_path.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"Path does not exist: {series_path}")
    return files


def load_and_sort_headers(files: list[Path]) -> list[Dataset]:
    """
    Load ONLY DICOM headers and sort by geometric Z (ImagePositionPatient[2]).
    """
    headers = [dcmread(str(fp), stop_before_pixels=True , defer_size="1 MB") for fp in files]

    try:
        headers.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    except Exception:
        # Some series may have missing InstanceNumber; fallback to filename order
        try:
            headers.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
        except Exception:
            pass
        
    return headers


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


def z_from_classic(ds_list) -> Optional[np.ndarray]:
    """
    Get per-slice physical Z using ImageOrientationPatient + ImagePositionPatient.
    Falls back to IPP[2]. Returns None if positions unavailable.
    """
    try:
        iop = np.asarray(ds_list[0].ImageOrientationPatient, dtype=np.float32)  # (6,)
        row, col = iop[:3], iop[3:]
        n = np.cross(row, col)  # slice normal
        zs = []
        for i, ds in enumerate(ds_list):
            ipp = np.asarray(
                getattr(ds, "ImagePositionPatient", [0, 0, i])[:3], dtype=np.float32
            )
            zs.append(float(np.dot(ipp, n)))
        return np.asarray(zs, dtype=np.float32)
    except Exception:
        try:
            return np.asarray(
                [float(ds.ImagePositionPatient[2]) for ds in ds_list], dtype=np.float32
            )
        except Exception:
            return None


def z_from_multiframe(ds) -> Optional[np.ndarray]:
    """
    Enhanced multi-frame: per-frame Z from PerFrameFunctionalGroupsSequence
    """
    try:
        pffg = ds.PerFrameFunctionalGroupsSequence
        zs = [float(fr.PlanePositionSequence[0].ImagePositionPatient[2]) for fr in pffg]
        return np.asarray(zs, dtype=np.float32)
    except Exception:
        return None


def _anti_alias_if_needed(vol: np.ndarray, target_z: int, enable: bool) -> np.ndarray:
    """
    Apply a light Gaussian blur along Z before downsampling to reduce aliasing
    (useful for very long CTA stacks). No-op if not downsampling.
    """
    z = vol.shape[0]
    if z <= target_z:
        return vol
    factor = z / float(target_z)
    sigma = 0.5 * min(max(factor - 1.0, 0.0), 2.0)
    if sigma <= 0:
        return vol
    # Only along Z (axis=0), 'nearest' matches edge behavior of medical stacks.
    return _gauss1d(vol, sigma=sigma, axis=0, mode="nearest")


def _resample_linear_spacing_aware(
    vol: np.ndarray, target_z: int, z_pos: Optional[np.ndarray]
) -> np.ndarray:
    """
    Core spacing-aware linear resampler along Z.
    - If z_pos provided, interpolate in physical space.
    - Else, interpolate in index space [0..Z-1].
    Returns float32 (target_z, H, W).
    """
    z, *_ = vol.shape
    if target_z is None or target_z == z:
        return vol.astype(np.float32, copy=False)

    if z <= 1:
        return np.repeat(vol, repeats=target_z, axis=0).astype(np.float32, copy=False)

    if z_pos is not None and len(z_pos) == z:
        # Sort by position and drop duplicates
        order = np.argsort(z_pos)
        zp = np.asarray(z_pos, np.float32)[order]
        vol = vol[order]
        keep = np.concatenate([[True], np.diff(zp) > 1e-6])
        zp, vol = zp[keep], vol[keep]
        z = len(zp)
        # Map target physical positions to fractional old indices
        new_pos = np.linspace(zp[0], zp[-1], target_z, dtype=np.float32)
        old_idx = np.arange(z, dtype=np.float32)
        frac = np.interp(new_pos, zp, old_idx)
    else:
        frac = np.linspace(0, z - 1, target_z, dtype=np.float32)

    i0 = np.floor(frac).astype(int)
    i1 = np.clip(i0 + 1, 0, z - 1)
    t = (frac - i0).reshape(-1, 1, 1).astype(np.float32)

    out = (1.0 - t) * vol[i0].astype(np.float32) + t * vol[i1].astype(np.float32)
    return out


def resample_z(
    vol: np.ndarray,
    target_z: int,
    z_pos: Optional[np.ndarray] = None,
    *,
    antialiasing: bool = False,
) -> np.ndarray:
    """
    Public API. Optional anti-alias (Gaussian) is applied only when downsampling.
    """
    vol = _anti_alias_if_needed(vol, target_z, enable=antialiasing)
    return _resample_linear_spacing_aware(vol, target_z, z_pos)


def load_series_ct(
    series_dir: str,
    target_slices: int,
    window: tuple[float, float] = FIXED_CT_WINDOW,
) -> np.ndarray:
    """
    Memory-efficient loads a CT/CTA series as (Z,H,W) float32 in [0,1].
    Always applies the fixed angiography window (center=300, width=1000) unless overridden.
    """
    files = list_dicom_files(Path(series_dir))
    headers = load_and_sort_headers(files)

    num_slices = len(headers)
    height = headers[0].Rows
    width = headers[0].Columns
    vol_hu = np.empty((num_slices, height, width), dtype=np.float32)

    full_ds_list = []

    for i, header in enumerate(headers):
        ds = dcmread(header.filename)
        vol_hu[i] = to_hu(ds, ds.pixel_array)
        full_ds_list.append(ds)

    wc, ww = window
    vol = window_hu(vol_hu, wc, ww)
    z_pos = z_from_classic(full_ds_list)

    return resample_z(vol, target_slices, z_pos=z_pos, antialiasing=True)


def load_series_mr(
    series_dir: str,
    target_slices: int,
    subtype: Optional[str] = None,
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
        ds = dcmread(str(files[0]))
        vol = ds.pixel_array.astype(np.float32)
        vol = np.asarray(vol)
        if vol.ndim == 2:
            vol = vol[None, :, :]
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

        # Try to sort frames by Z using Per-frame Functional Groups
        try:
            pffg = ds.PerFrameFunctionalGroupsSequence
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

        z_pos = z_from_multiframe(ds)
        return resample_z(vol, target_slices, z_pos=z_pos, antialiasing=False)

    # Classic MR: load & sort all slices (with pixels), then normalize
    ds_list = load_slices(files)
    if subtype == "MRA":
        sl = [apply_voi_or_minmax(ds, ds.pixel_array) for ds in ds_list]
        vol = np.stack(sl, axis=0).astype(np.float32)
    else:
        vol = np.stack([ds.pixel_array.astype(np.float32) for ds in ds_list], axis=0)
        vol = zscore_to_unit(vol)

    z_pos = z_from_classic(ds_list)
    return resample_z(vol, target_slices, z_pos=z_pos, antialiasing=False)


def load_series_auto(
    series_dir: str,
    target_slices: int,
    subtype: Optional[str] = None,
) -> np.ndarray:
    if subtype in {"CT", "CTA"}:
        return load_series_ct(
            series_dir, target_slices=target_slices, window=FIXED_CT_WINDOW
        )
    return load_series_mr(series_dir, subtype=subtype, target_slices=target_slices)
