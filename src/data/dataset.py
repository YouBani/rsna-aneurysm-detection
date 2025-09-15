import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from .utils import safe_float, load_series_auto

from typing import Optional


class RSNADataset(Dataset):
    """
    PyTorch Dataset for a 3D DICOM classification.

    Returns a dictionary for each sample:
      {
        "id":      str (SeriesInstanceUID),
        "image":   FloatTensor (1, Z, H, W) in [0,1],
        "label":   FloatTensor (),  # 0.0 / 1.0
        "age":     FloatTensor (),  # years, -1.0 if unknown
        "sex":     FloatTensor (),  # {M:0, F:1, unknown:-1}
        "weight":  FloatTensor ()   # kg, -1.0 if unknown

      }
    """

    def __init__(
        self,
        jsonl_path: str,
        target_slices=125,
        cache_dir: Optional[str] = None,
        transform=None,
    ):
        self.rows = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines()]
        self.target_slices = target_slices
        self.cache = Path(cache_dir) if cache_dir else None
        self.transform = transform
        if cache_dir:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _parse_sex(v) -> float:
        if not isinstance(v, str):
            return -1.0
        v = v.upper()
        if v == "MALE":
            return 0.0
        if v == "FEMALE":
            return 1.0
        return -1.0

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        series_uid = row["id"]
        image_path = Path(row["image_path"])

        vol = None
        cache_path = self.cache / f"{series_uid}.npy" if self.cache else None

        if cache_path and cache_path.exists():
            try:
                vol = np.load(cache_path)
            except Exception as e:
                print(f"[WARN]: Failed to load cached file {cache_path}: {e}")
                vol = None

        if vol is None:
            vol = load_series_auto(
                str(image_path),
                subtype=row["subtype"],
                target_slices=self.target_slices,
            )

            if not isinstance(vol, np.ndarray):
                raise TypeError(
                    f"load_series_auto must return np.ndarray, got {type(vol)} for {series_uid}"
                )

            if cache_path:
                np.save(cache_path, vol)

        sex = self._parse_sex(row.get("patient_sex"))
        age = safe_float(row.get("patient_age"), -1.0)
        weight = safe_float(row.get("patient_weight"), -1.0)

        x = torch.from_numpy(vol).unsqueeze(0).float()
        if self.transform:
            x = self.transform(x)

        return {
            "id": series_uid,
            "image": x,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "age": torch.tensor(age, dtype=torch.float32),
            "sex": torch.tensor(sex, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
        }
