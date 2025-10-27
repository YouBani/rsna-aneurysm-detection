# rsna-aneurysm-detection

## Overview

This project was developed for the *RSNA Intracranial Aneurysm Detection Challenge*, which involves detecting aneurysms in 3D CT angiography scans (DICOM format).
The main goal was to build an end-to-end 3D convolutional pipeline capable of handling large volumetric data efficiently on limited GPU memory.

*Key Features*
* Patient-level stratified splits to prevent leakage
* Offline preprocessing to `.npy` for fast I/O
* 3D ResNet-18 for volumetric aneurysm classification
* AWS cloud training with FSx Lustre and S3 integration
* Weights & Biases for experiment tracking
* Gradient checkpointing and offline preprocessing to reduce memory bottlenecks

## Project Structure
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

