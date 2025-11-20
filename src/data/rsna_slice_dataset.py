import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Any
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(image_size: int, is_train: bool) -> A.Compose:
    """Returns a set of Albumentations transforms for 2D classification."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )


class RSNASliceDataset(Dataset):
    """
    PyTorch Dataset for loading individual 2.5D slices from preprocessed .npy files.
    Reads from a manifest created by `build_slice_manifest.py`.

    Returns:
        image: (3, H, W) tensor
        target: (1,) tensor. 0.0 or 1.0 (slice-level label)
        series_id: str
        slice_idx: int
    """

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[A.Compose] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform

        self.manifest: list[dict[str, Any]] = []
        with open(self.manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["slice_idx"] = int(row["slice_idx"])
                row["label"] = float(row["label"])
                self.manifest.append(row)
        self.data_cache: dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def _get_memmapped_array(self, path: str) -> np.memmap:
        """Uses mmap_mode='r' for fast reading of .npy files."""
        if path not in self.data_cache:
            self.data_cache[path] = np.load(path, mmap_mode="r")
        return self.data_cache[path]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest[idx]

        series_id = row["series_id"]
        vol_path = row["vol_path"]
        slice_idx = row["slice_idx"]

        vol = self._get_memmapped_array(vol_path)
        num_slices = vol.shape[0]

        img_c = vol[slice_idx].astype(np.float32)
        img_p = vol[max(0, slice_idx - 1)].astype(np.float32)
        img_n = vol[min(num_slices - 1, slice_idx + 1)].astype(np.float32)

        image_stack = np.dstack([img_p, img_c, img_n])

        target = torch.tensor(float(row.get("label", 0.0)), dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image_stack)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(np.transpose(image_stack, (2, 0, 1)))

        return {
            "image": image_tensor,
            "target": target.unsqueeze(0),
            "series_id": series_id,
            "slice_idx": slice_idx,
        }
