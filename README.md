# rsna-aneurysm-detection

## Overview

This project was developed for the *RSNA Intracranial Aneurysm Detection Challenge*, which involves detecting aneurysms in 3D CT angiography scans (DICOM format).
The main goal was to build an end-to-end 3D convolutional pipeline capable of handling large volumetric data efficiently on limited GPU memory.

*Key Features*
* 3D ResNet-18 for volumetric aneurysm classification
* MONAI-based preprocessing and augmentation
* AWS cloud training with FSx Lustre and S3 integration
* Weights & Biases for experiment tracking
* Gradient checkpointing and offline preprocessing to reduce memory bottlenecks

## Project Structure
