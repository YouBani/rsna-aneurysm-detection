import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Any
from .utils import safe_float
from src.constants.rsna import JSONL_LABEL_KEYS, PRESENT_IDX


class RSNADataset(Dataset):
    """
    PyTorch Dataset for a 3D DICOM classification.

    Returns a dictionary for each sample:
      {
        "id":        str (SeriesInstanceUID),
        "image":     FloatTensor (1, Z, H, W) in [0,1],
        "label":     FloatTensor (),  # 0.0 / 1.0
        "labels14":  FloatTensor (14,),
        "age":       FloatTensor (),  # years, -1.0 if unknown
        "sex":       FloatTensor (),  # {M:0, F:1, unknown:-1}
        "weight":    FloatTensor ()   # kg, -1.0 if unknown
        "bbox":      FloatTensor (4,) # (y0, y1, x0, x1)
        "z_pos":     FloatTensor (Z,) # Physical Z positions
      }
    """

    def __init__(
        self,
        preprocessed_dir: Path,
        manifest_rows: list[dict[str, Any]],
        transform=None,
    ):
        self.preprocessed_dir = preprocessed_dir
        self.rows = manifest_rows
        self.transform = transform

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

    @staticmethod
    def _labels14_from_row(row: dict) -> Optional[torch.Tensor]:
        vals = []
        try:
            for key in JSONL_LABEL_KEYS:
                vals.append(float(row[key]))
        except KeyError:
            return None
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        series_uid = row["id"]

        npz_path = self.preprocessed_dir / f"{series_uid}.npz"

        try:
            with np.load(npz_path) as data:
                vol = data["vol"].astype(np.float32)
                bbox = data.get("bbox", np.array([-1, -1, -1, -1]))
                z_pos = data.get("z_pos", np.array([]))
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing preprocessed file: {npz_path}")
        except Exception as e:
            raise IOError(f"Failed to load {npz_path}: {e}")

        sex = self._parse_sex(row.get("patient_sex"))
        age = safe_float(row.get("patient_age"), -1.0)
        weight = safe_float(row.get("patient_weight"), -1.0)

        x = torch.from_numpy(vol).unsqueeze(0).float()

        if self.transform:
            x = self.transform(x)

        out = {
            "id": series_uid,
            "image": x,
            "label": torch.tensor(float(row.get("label", 0.0)), dtype=torch.float32),
            "age": torch.tensor(age, dtype=torch.float32),
            "sex": torch.tensor(sex, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "bbox": torch.from_numpy(bbox).int(),
            "z_pos": torch.from_numpy(z_pos).float(),
        }

        labels14 = self._labels14_from_row(row)
        if labels14 is not None:
            out["labels14"] = labels14
            out["label"] = labels14[PRESENT_IDX]

        return out
