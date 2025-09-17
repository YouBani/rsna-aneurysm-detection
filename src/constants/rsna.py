LABEL_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]

JSONL_LABEL_KEYS = [
    "left_infraclinoid_ica",
    "right_infraclinoid_ica",
    "left_supraclinoid_ica",
    "right_supraclinoid_ica",
    "left_middle_cerebral_artery",
    "right_middle_cerebral_artery",
    "anterior_communicating_artery",
    "left_anterior_cerebral_artery",
    "right_anterior_cerebral_artery",
    "left_posterior_communicating_artery",
    "right_posterior_communicating_artery",
    "basilar_tip",
    "other_posterior_circulation",
    "label",
]


PRESENT_IDX = len(JSONL_LABEL_KEYS) - 1
K = len(JSONL_LABEL_KEYS)
