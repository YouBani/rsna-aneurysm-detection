import re
from pydicom.dataset import Dataset


def get_upper_str(ds: Dataset, attr: str) -> str:
    """Safely gets an attribute from a pydicom dataset and returns it as an uppercase string."""
    return str(getattr(ds, attr, "") or "").upper()


def infer_subtype(ds_first: Dataset) -> str:
    """
    Infers a broad imaging subtype by analyzing DICOM metadata.

    Args:
        ds_first: The first pydicom.dataset.Dataset object from a given series.

    Returns:
        A string representing the inferred subtype. Examples include:
        "CTA", "CT", "MRA", "MRI T1post", "MRI T2", "MR", or "Unknown".
    """
    mod = get_upper_str(ds_first, "Modality")
    desc = (
        get_upper_str(ds_first, "SeriesDescription")
        + " "
        + get_upper_str(ds_first, "ProtocolName")
    )

    try:
        proc_code = ""
        for item in getattr(ds_first, "ProcedureCodeSequence", []) or []:
            proc_code += get_upper_str(item, "CodeMeaning") + " "
        if any(k in proc_code for k in ["ANGIO", "ANJIYOGRAFI", "MRA"]):
            return "CTA" if mod == "CT" else "MRA"
    except Exception:
        pass

    cta_keywords = ["CTA", "ANGIO", "CAROTID", "MIP", "AX-MIP"]
    mra_keywords = ["MRA", "ANGIO", "TOF", "3DTOF", "CEMRA", "FL3D", "SPGR"]
    t1_tokens = ["T1", "MPRAGE", "BRAVO", "SPGR", "FSPGR", "MPR"]
    t1_post_tokens = ["POST", "+C", "GAD", "GD", "CONTRAST"]

    if mod == "CT":
        if any(k in desc for k in cta_keywords):
            return "CTA"
        return "CT"

    if mod == "MR":
        if any(k in desc for k in mra_keywords) or re.search(r"\bCOW\b", desc):
            return "MRA"

        if any(k in desc for k in t1_tokens) and any(k in desc for k in t1_post_tokens):
            return "MRI T1post"

        if "T2" in desc:
            return "MRI T2"

        return "MR"

    return mod if mod else "Unknown"
