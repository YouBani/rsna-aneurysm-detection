from pathlib import Path
import pandas as pd

def build_manifest(data_root: str) -> list[dict]:
    """
    Builds a list of dictionaries for the RSNA aneurysm detection dataset.

    Args:
        data_root (str): The root directory of the dataset.
    
    Returns:
        list[dict]: A list of dictionaries, one per SeriesInstanceUID.
    """
    root = Path(data_root)
    raw_data = root / "raw"
    csv_path = raw_data / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data directory not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    manifest = []
    for _, row in df.iterrows():
        series_uid = row['SeriesInstanceUID']
        image_path = raw_data / "series" / series_uid
        seg_path = raw_data / "segmentations" / f"{series_uid}.nii"

        entry = {
        'id': series_uid,
        'image_path': str(image_path),
        'segmentation_path': str(seg_path) if seg_path.exists() else None,
        'label': int(row['Aneurysm Present']),
        'patient_age': int(row['PatientAge']),
        'patient_sex': row['PatientSex'],

        'left_infraclinoid_ica': int(row['Left Infraclinoid Internal Carotid Artery']),
        'right_infraclinoid_ica': int(row['Right Infraclinoid Internal Carotid Artery']),
        'left_supraclinoid_ica': int(row['Left Supraclinoid Internal Carotid Artery']),
        'right_supraclinoid_ica': int(row['Right Supraclinoid Internal Carotid Artery']),
        'left_middle_cerebral_artery': int(row['Left Middle Cerebral Artery']),
        'right_middle_cerebral_artery': int(row['Right Middle Cerebral Artery']),
        'anterior_communicating_artery': int(row['Anterior Communicating Artery']),
        'left_anterior_cerebral_artery': int(row['Left Anterior Cerebral Artery']),
        'right_anterior_cerebral_artery': int(row['Right Anterior Cerebral Artery']),
        'left_posterior_communicating_artery': int(row['Left Posterior Communicating Artery']),
        'right_posterior_communicating_artery': int(row['Right Posterior Communicating Artery']),
        'basilar_tip': int(row['Basilar Tip']),
        'other_posterior_circulation': int(row['Other Posterior Circulation'])
        }
        manifest.append(entry)

    return manifest