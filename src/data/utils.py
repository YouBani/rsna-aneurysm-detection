from pathlib import Path
from typing import Optional, Literal, Any

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixel_data_handlers.util import apply_voi_lut
from scipy.ndimage import gaussian_filter1d as _gauss1d
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_erosion


def safe_float(value, default: float = -1.0) -> float:
    """Safely converts a value to a float, handling None or errors."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _list_dicom_files(series_path: Path) -> list[Path]:
    """Return all *.dcm files in a series folder."""
    files = sorted(series_path.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"Path does not exist: {series_path}")
    return files


def _sort_headers_by_z(headers: list[Dataset]) -> list[Dataset]:
    """
    Sort by Z (ImagePositionPatient[2]), fallback to InstanceNumber.
    """
    try:
        headers.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    except Exception:
        try:
            headers.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
        except Exception:
            pass

    return headers


def _is_multiframe(ds: Dataset) -> bool:
    """Enhanced MR often stores an entire stack as one multi-frame DICOM."""
    return hasattr(ds, "NumberOfFrames")


def _load_series_pixels(files: list[Path]) -> tuple[np.ndarray, list[Dataset]]:
    """
    Classic series: return (Z, H, W) float32 from stacked 2D slices and list of full dataset.
    """
    headers = [
        dcmread(str(fp), stop_before_pixels=True, defer_size="1 MB") for fp in files
    ]
    headers = _sort_headers_by_z(headers)

    vol = []
    full = []
    for h in headers:
        ds = dcmread(h.filename)
        vol.append(ds.pixel_array.astype(np.float32))
        full.append(ds)
    vol = np.stack(vol, axis=0)
    return vol, full


def _load_multiframe_pixels(fp: Path) -> tuple[np.ndarray, Dataset]:
    ds = dcmread(str(fp))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    try:
        pffg = ds.PerFrameFunctionalGroupsSequence
        z = [float(fr.PlanePositionSequence[0].ImagePositionPatient[2]) for fr in pffg]
        order = np.argsort(z)
        arr = arr[order]
    except Exception:
        pass
    return arr, ds


def to_hu(ds, arr: np.ndarray) -> np.ndarray:
    """CT: convert raw pixels to Hounsfield Units using slope/intercept."""
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr.astype(np.float32) * slope + intercept


def window_hu(vol: np.ndarray, center: float, width: float) -> np.ndarray:
    """Perform windowing operation by Clipping HU to [C - W/2, C + W/2] and scaling to [0, 1]."""
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


def _z_from_classic(headers: list[Dataset]) -> Optional[np.ndarray]:
    """
    Get per-slice physical Z using ImageOrientationPatient + ImagePositionPatient.
    Falls back to IPP[2]. Returns None if positions unavailable.
    """
    try:
        iop = np.asarray(headers[0].ImageOrientationPatient, dtype=np.float32)  # (6,)
        row, col = iop[:3], iop[3:]
        n = np.cross(row, col)  # slice normal
        zs = []
        for i, ds in enumerate(headers):
            ipp = np.asarray(
                getattr(ds, "ImagePositionPatient", [0, 0, i])[:3], dtype=np.float32
            )
            zs.append(float(np.dot(ipp, n)))
        return np.asarray(zs, dtype=np.float32)
    except Exception:
        try:
            return np.asarray(
                [float(ds.ImagePositionPatient[2]) for ds in headers], dtype=np.float32
            )
        except Exception:
            return None


def _z_from_multiframe(ds) -> Optional[np.ndarray]:
    """Extracts the Z-position for each frame from multiframe DICOM."""
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


def _largest_cc_mask_2d(mask2d: np.ndarray) -> np.ndarray:
    """Keep only the largest 2D connected component0"""
    if mask2d.dtype != bool:
        mask2d = mask2d.astype(bool)
    if not mask2d.any():
        return mask2d
    lbl, n = cc_label(mask2d)
    if n <= 1:
        return mask2d
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    return lbl == keep


def _apply_central_filter(mask2d: np.ndarray, band_frac: float = 0.9) -> np.ndarray:
    """Intersect mask with a central band to remove edge artifacts."""
    h, w = mask2d.shape
    cy0 = int((1 - band_frac) * 0.5 * h)
    cy1 = int(h - cy0)
    cx0 = int((1 - band_frac) * 0.5 * w)
    cx1 = int(w - cx0)
    central = np.zeros_like(mask2d, dtype=bool)
    central[cy0:cy1, cx0:cx1] = True
    return mask2d & central if mask2d.any() else central


def _bbox_from_mask(
    mask2d: np.ndarray, pad: int, h: int, w: int
) -> tuple[int, int, int, int]:
    """Get (y0, y1, x0, x1) from a 2D mask."""
    if not mask2d.any():
        return 0, h, 0, w
    ys = np.where(mask2d.any(axis=1))[0]
    xs = np.where(mask2d.any(axis=0))[0]
    y0 = max(int(ys[0] - pad), 0)
    y1 = min(int(ys[-1] + 1 + pad), h)
    x0 = max(int(xs[0] - pad), 0)
    x1 = min(int(xs[-1] + 1 + pad), w)
    return y0, y1, x0, x1


def crop_and_normalize_ct(
    series_dir: str | Path,
    *,
    hu_threshold: float = -300.0,
    silhouette_min_frac: float = 0.02,
    central_band_frac: float = 0.90,
    pad: int = 8,
    window_center: float = 300.0,
    window_width: float = 1000.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Loads, normalizes, and crops a CT series.

    Args:
        series_dir: Path to the folder containing DICOM files.
        hu_threshold: HU value used to find the brain silhouette.
        silhouette_min_frac: Minimum fraction of slices a pixel must
            appear in to be part of the 2D silhouette.
        central_band_frac: Fraction of the center to keep when filtering
            the silhouette to remove bounding box.
        pad: Pixels of padding to add to the final bounding box.
        window_center: The center of the HU window.
        window_width: The width of the HU window.

    Returns:
        vol_final: (target_z, Hc, Wc) float32 in [0,1]
        meta: A dictionary containing 'bbox' and 'z_pos'
    """
    files = _list_dicom_files(Path(series_dir))
    z_pos: Optional[np.ndarray] = None

    if _is_multiframe(dcmread(str(files[0]), stop_before_pixels=True)):
        vol_raw, ds = _load_multiframe_pixels(files[0])
        vol_hu = to_hu(ds, vol_raw)
        z_pos = _z_from_multiframe(ds)
    else:
        vol_raw, full = _load_series_pixels(files)
        vol_hu = np.stack([to_hu(ds, sl) for ds, sl in zip(full, vol_raw)], axis=0)
        z_pos = _z_from_classic(full)

    z, h, w = vol_hu.shape

    # 1. Crop of raw HU
    mask3d = vol_hu > hu_threshold
    sil = mask3d.mean(axis=0) > silhouette_min_frac
    sil = _largest_cc_mask_2d(sil)
    sil = _apply_central_filter(sil, band_frac=central_band_frac)

    if sil.sum() < 64:
        relaxed = ~binary_erosion(~sil, iterations=2)
        sil = relaxed | sil

    y0, y1, x0, x1 = _bbox_from_mask(sil, pad=pad, h=h, w=w)
    vol_crop_hu = vol_hu[:, y0:y1, x0:x1]

    # 2. Normalize
    vol_norm = window_hu(vol_crop_hu, center=window_center, width=window_width)

    meta = {
        "bbox": (int(y0), int(y1), int(x0), int(x1)),
        "z_pos": z_pos,
    }
    return vol_norm, meta


def crop_and_normalize_mr(
    series_dir: str | Path,
    *,
    subtype: Optional[Literal["MRA", "T1", "T2", "MR"]] = None,
    norm_clamp: float = 5.0,
    crop_threshold: Optional[float] = None,
    thr_percentile: float = 1.0,
    silhouette_min_frac: float = 0.02,
    central_band_frac: float = 0.90,
    pad: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Loads, normalizes and crops an MR or MRA series.

    Args:
        series_dir: Path to the folder containing DICOM files.
        subtype: The type of MR scan. 'MRA' uses per-slice VOI/min-max,
            while 'MR', 'T1', 'T2' use volume-wise z-scoring.
        norm_clamp: For z-score normalization, the value to clamp at.
            (e.g., 5.0 means clamp to ±5 standard deviations).
        crop_threshold: Manual cutoff value (0-1) to create the silhouette
        mask. If None, the threshold is auto-calculated using
        `thr_percentile`.
        thr_percentile: The percentile (e.g., 1.0) to use as a base for
            auto-calculating the `crop_threshold`.
        silhouette_min_frac: Minimum fraction of slices a pixel must
            appear in to be part of the 2D silhouette.
        central_band_frac: Fraction of the center to keep when filtering
            the silhouette to remove bounding box.
        pad: Pixels of padding to add to the final bounding box.

    Returns:
        vol_final: (target_z, Hc, Wc) float32 in [0,1]
        meta: A dictionary containing 'bbox' and 'z_pos'
    """
    files = _list_dicom_files(Path(series_dir))
    first_header = dcmread(str(files[0]), stop_before_pixels=True)
    z_pos: Optional[np.ndarray] = None
    vol_norm: Optional[np.ndarray] = None
    h, w = first_header.Rows, first_header.Columns

    if _is_multiframe(first_header):
        vol_raw, ds = _load_multiframe_pixels(files[0])
        _, h, w = vol_raw.shape

        # 1. Normalize
        if subtype == "MRA":
            vol_norm = np.stack([apply_voi_or_minmax(ds, sl) for sl in vol_raw], axis=0)
        else:
            vol_norm = zscore_to_unit(vol_raw, clamp=norm_clamp)
        z_pos = _z_from_multiframe(ds)
    else:
        vol_slices, full = _load_series_pixels(files)
        _, h, w = vol_slices.shape
        # 1. Normalize
        if subtype == "MRA":
            vol_norm = np.stack(
                [apply_voi_or_minmax(ds, sl) for ds, sl in zip(full, vol_slices)],
                axis=0,
            )
        else:
            vol_norm = zscore_to_unit(vol_slices, clamp=norm_clamp)
        z_pos = _z_from_classic(full)

    if vol_norm is None:
        raise ValueError("Volume normalization failed.")

    vol_norm = vol_norm.astype(np.float32, copy=False)

    # 2. Crop
    if crop_threshold is None:
        base = np.percentile(vol_norm, thr_percentile)
        span = float(vol_norm.max() - vol_norm.min())
        threshold = base + 0.05 * span
        threshold = float(np.clip(threshold, 0.05, 0.2))
    else:
        threshold = float(crop_threshold)

    mask3d = vol_norm > threshold
    sil = mask3d.mean(axis=0) > silhouette_min_frac
    sil = _largest_cc_mask_2d(sil)
    sil = _apply_central_filter(sil, band_frac=central_band_frac)

    if sil.sum() < 64:
        relaxed = ~binary_erosion(~sil, iterations=2)
        sil = relaxed | sil

    y0, y1, x0, x1 = _bbox_from_mask(sil, pad=pad, h=h, w=w)
    vol_crop = vol_norm[:, y0:y1, x0:x1]

    meta = {
        "bbox": (int(y0), int(y1), int(x0), int(x1)),
        "z_pos": z_pos,
    }
    return vol_crop, meta


def crop_and_normalize_series(
    series_dir: str | Path,
    modality: Literal["CT", "CTA", "MR", "MRI", "MRA"],
    **kwargs,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Public API to load, crop, and normalize a DICOM series.

    Args:
        series_dir: Path to the folder containing DICOM files.
        modality: one of {"CT","CTA","MR","MRI","MRA"}.
        **kwargs:
            CT:  hu_threshold, silhouette_min_frac, central_band_frac, pad,
                 window_center, window_width
            MR:  subtype, norm_clamp, crop_threshold, thr_percentile,
                 silhouette_min_frac, central_band_frac, pad

    Returns:
        A tupe of (vol_final, bbox):
        - vol_final(np.ndarray): The processed(cropped, normalized, resampled)
            volume as a float32 array in [0, 1] with shape
            (target_z, H_cropped, W_cropped)
        - meta: A dictionary containing 'bbox' and 'z_pos'
    """
    if modality in {"CT", "CTA"}:
        return crop_and_normalize_ct(series_dir, **kwargs)

    elif "MR" in modality:
        if modality == "MRA":
            subtype = "MRA"
        else:
            subtype = "MR"
        return crop_and_normalize_mr(series_dir, subtype=subtype, **kwargs)
    else:
        raise ValueError(f"Unknow modality: {modality}")
