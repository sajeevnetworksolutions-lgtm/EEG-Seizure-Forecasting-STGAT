"""
EDF Recording Ingestion Module using MNE-Python
"""

from pathlib import Path
import numpy as np

def load_edf(path: str | Path, preload: bool = True):
    """
    Load an EDF EEG recording using MNE-Python.
    
    Parameters
    ----------
    path : str | Path
        Path to the .edf file.
    preload : bool
        Whether to load the raw signal data into memory.
        
    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object containing the multi-channel EEG recording.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path.resolve()}")
        
    try:
        if hasattr(np, "dtypes") and not hasattr(np.dtypes, "StringDType"):
            np.dtypes.StringDType = np.dtype("U")
        import mne
    except ImportError:
        raise ImportError("MNE-Python is required to load EDF files. Run: pip install mne")
        
    raw = mne.io.read_raw_edf(
        path,
        preload=preload,
        verbose="ERROR"
    )
    return raw

def get_channel_mapping(raw_ch_names, target_channels):
    """
    Construct mapping from recording's channel names (including bipolar pairs)
    to standardized 10-20 electrode names.
    """
    mapping = {}
    for target in target_channels:
        # Search for exact match or bipolar match (e.g. 'FP1-F7' for 'FP1', 'C3-P3' for 'C3')
        target_upper = target.upper()
        matches = [c for c in raw_ch_names if target_upper in c.upper().split('-') or target_upper == c.upper().replace('EEG ', '').strip()]
        if matches:
            mapping[target] = matches[0]
        else:
            # Fallback substring search
            fallback = [c for c in raw_ch_names if target_upper in c.upper()]
            if fallback:
                mapping[target] = fallback[0]
    return mapping
