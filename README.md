# RSNA Intracranial Aneurysm Detection

## 1. Overview

This project was developed for the *RSNA Intracranial Aneurysm Detection Challenge*, which involves detecting aneurysms in 3D CT angiography scans (DICOM format).
The main goal was to build an end-to-end 3D convolutional pipeline capable of handling large volumetric data efficiently on limited GPU memory.

*Key Features*
* Patient-level stratified splits to prevent leakage
* Offline preprocessing to `.npy` for fast I/O
* 3D ResNet-18 for volumetric aneurysm classification
* AWS cloud training with FSx Lustre and S3 integration
* Weights & Biases for experiment tracking
* Gradient checkpointing and offline preprocessing to reduce memory bottlenecks

---

## 2. Dataset
The dataset originates from the **RSNA Intracranial Aneurysm Detection Challenge**.

**Content:**
- 3,400 3D CTA and MRA studies in DICOM format  
- Associated CSV file with patient metadata and 14 binary labels:  
  - 13 vessel-level aneurysm indicators  
  - 1 global `"Aneurysm Present"` label

**Modalities handled:**
- `CT`, `CTA` (Computed Tomography Angiography)
- `MR`, `MRA` (Magnetic Resonance Angiography)
- Including variants such as `MRI T1post`, `MRI T2`

---

## 3. Technical Pipeline

### 🧮 4.1 Data Preprocessing
Implemented in `src/data/preprocess.py`, the offline preprocessing pipeline:
- Reads **DICOM headers** and infers scan subtype (`CT/CTA/MRA/T1post/T2`)
- Converts pixel data to **Hounsfield Units** for CT
- Applies **VOI LUT** or **z-score normalization** for MR
- Resamples the Z-dimension with **physical spacing awareness**
- Saves final volumes as `.npy` arrays (float32, normalized [0,1])  

This offline conversion eliminates runtime I/O overhead and minimizes RAM usage during training.

---

### 4.2 Dataset Manifest
`build_manifest.py` consolidates patient metadata, file paths, modality, subtype, and 14 binary labels into a single `.jsonl` manifest:
```json
{
  "id": "1.2.840....",
  "modality": "CTA",
  "subtype": "CTA",
  "patient_id": "RSNA_0001",
  "patient_age": 63,
  "patient_sex": "FEMALE",
  "patient_weight": 64.2,
  "image_path": "data/raw/series/1.2.840...",
  "label": 1,
  "left_middle_cerebral_artery": 1,
  "right_middle_cerebral_artery": 0,
  ...
}
```

### 4.3 Dataset & Loader

RSNADataset (in dataset.py):

Loads cached .npy volumes

Normalizes metadata (age, sex, weight)

Returns a dictionary for each sample:

{
    "image": Tensor (1, Z, H, W),
    "labels14": Tensor (14,),
    "age": float,
    "sex": float,
    "weight": float
}


`data.py` wraps this into PyTorch `DataLoader` objects with optional weighted sampling for class imbalance and deterministic seeding.

### 4.4 Model Architecture

Implemented in `src/models/model.py`:

* Base: 3D ResNet-18 (torchvision.models.video.r3d_18)

* Input: single-channel 3D tensor (1, Z, H, W)

* Output: 14 sigmoid logits (multi-label)

* Options:

    * Replace BatchNorm with GroupNorm
    * Enable gradient checkpointing for deep feature stages
    * Mixed precision training (fp16 / bf16)
```python
model = build_3d_model(
    in_channels=1,
    num_outputs=14,
    checkpointing=True,
    use_groupnorm=True
)

```
## Project Structure
```
rsna-aneurysm-detection/
└── src/
    ├── constants/
    │   └── rsna.py                # Column names and JSONL label keys
    ├── data/
    │   ├── __init__.py
    │   ├── build_manifest.py      # Build JSONL manifest from RSNA CSV + DICOM headers
    │   ├── data.py                # DataLoader builder + weighted sampling
    │   ├── dataset.py             # RSNADataset: returns (1, Z, H, W) float tensor + meta
    │   ├── preprocess.py          # Offline preprocessing to .npy with multiprocessing
    │   ├── series_classifier.py   # Infer subtype (CT/CTA/MR/MRA/T1post/T2) from metadata
    │   ├── split.py               # Patient-level stratified train/val/test split
    │   └── utils.py               # DICOM I/O, normalization, HU/VOI-LUT, z-resampling, helpers
    ├── metrics/
    │   └── rsna.py                              # MultilabelAUROC + weighted AUC helper
    ├── models/
    │   ├── __init__.py
    │   └── model.py                             # 3D ResNet-18, BN→GN option, checkpointing wrapper
    └── trainer/
        ├── __init__.py
        ├── act_hooks.py                  # activation stats, sampling, hist buffers
        ├── loops.py                      # train_one_epoch / validate for 14-label setup
        ├── main.py                       # CLI entrypoint (args, W&B, loaders, scheduler)
        ├── train.py                      # high-level training orchestration + checkpoints
        └── utils.py                      # seeding, pos_weight calc, metrics, CUDA helpers

```
