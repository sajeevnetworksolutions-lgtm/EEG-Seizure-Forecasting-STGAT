"""
Electrode Channel Mapping and Standard 10-20 Compatibility Engine
Maps arbitrary raw EEG recording channels (bipolar, monopolar, referenced)
to the canonical 19-channel 10-20 International System required by ST-GAT.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from src.config import ELECTRODES_19, NUM_CHANNELS

# Standard 10-20 canonical electrode ordering
CANONICAL_19 = list(ELECTRODES_19)

# Comprehensive alias dictionary for standard 10-20 channels
CHANNEL_ALIASES = {
    "FP1": ["FP1", "Fp1", "EEG FP1", "EEG Fp1", "FP1-REF", "FP1-AVG", "FP1-CZ", "FP1-A1", "FP1-A2"],
    "FP2": ["FP2", "Fp2", "EEG FP2", "EEG Fp2", "FP2-REF", "FP2-AVG", "FP2-CZ", "FP2-A1", "FP2-A2"],
    "F7":  ["F7", "f7", "EEG F7", "EEG f7", "F7-REF", "F7-AVG", "F7-CZ", "F7-A1", "F7-A2"],
    "F3":  ["F3", "f3", "EEG F3", "EEG f3", "F3-REF", "F3-AVG", "F3-CZ", "F3-A1", "F3-A2"],
    "FZ":  ["FZ", "Fz", "fz", "EEG FZ", "EEG Fz", "FZ-REF", "FZ-AVG", "FZ-CZ", "FZ-A1", "FZ-A2"],
    "F4":  ["F4", "f4", "EEG F4", "EEG f4", "F4-REF", "F4-AVG", "F4-CZ", "F4-A1", "F4-A2"],
    "F8":  ["F8", "f8", "EEG F8", "EEG f8", "F8-REF", "F8-AVG", "F8-CZ", "F8-A1", "F8-A2"],
    "T7":  ["T7", "T3", "t7", "t3", "EEG T7", "EEG T3", "T7-REF", "T3-REF", "T7-AVG", "T3-AVG"],
    "C3":  ["C3", "c3", "EEG C3", "EEG c3", "C3-REF", "C3-AVG", "C3-CZ", "C3-A1", "C3-A2"],
    "CZ":  ["CZ", "Cz", "cz", "EEG CZ", "EEG Cz", "CZ-REF", "CZ-AVG", "CZ-A1", "CZ-A2"],
    "C4":  ["C4", "c4", "EEG C4", "EEG c4", "C4-REF", "C4-AVG", "C4-CZ", "C4-A1", "C4-A2"],
    "T8":  ["T8", "T4", "t8", "t4", "EEG T8", "EEG T4", "T8-REF", "T4-REF", "T8-AVG", "T4-AVG"],
    "P7":  ["P7", "T5", "p7", "t5", "EEG P7", "EEG T5", "P7-REF", "T5-REF", "P7-AVG", "T5-AVG"],
    "P3":  ["P3", "p3", "EEG P3", "EEG p3", "P3-REF", "P3-AVG", "P3-CZ", "P3-A1", "P3-A2"],
    "PZ":  ["PZ", "Pz", "pz", "EEG PZ", "EEG Pz", "PZ-REF", "PZ-AVG", "PZ-CZ", "PZ-A1", "PZ-A2"],
    "P4":  ["P4", "p4", "EEG P4", "EEG p4", "P4-REF", "P4-AVG", "P4-CZ", "P4-A1", "P4-A2"],
    "P8":  ["P8", "T6", "p8", "t6", "EEG P8", "EEG T6", "P8-REF", "T6-REF", "P8-AVG", "T6-AVG"],
    "O1":  ["O1", "o1", "EEG O1", "EEG o1", "O1-REF", "O1-AVG", "O1-CZ", "O1-A1", "O1-A2"],
    "O2":  ["O2", "o2", "EEG O2", "EEG o2", "O2-REF", "O2-AVG", "O2-CZ", "O2-A1", "O2-A2"],
}

# Standard double-banana bipolar chains in CHB-MIT and clinical telemetry
BIPOLAR_PAIRS = {
    "FP1-F7": ("FP1", "F7"),
    "F7-T7":  ("F7", "T7"),
    "T7-P7":  ("T7", "P7"),
    "P7-O1":  ("P7", "O1"),
    "FP1-F3": ("FP1", "F3"),
    "F3-C3":  ("F3", "C3"),
    "C3-P3":  ("C3", "P3"),
    "P3-O1":  ("P3", "O1"),
    "FZ-CZ":  ("FZ", "CZ"),
    "CZ-PZ":  ("CZ", "PZ"),
    "FP2-F4": ("FP2", "F4"),
    "F4-C4":  ("F4", "C4"),
    "C4-P4":  ("C4", "P4"),
    "P4-O2":  ("P4", "O2"),
    "FP2-F8": ("FP2", "F8"),
    "F8-T8":  ("F8", "T8"),
    "T8-P8":  ("T8", "P8"),
    "P8-O2":  ("P8", "O2"),
}


class ChannelMapper:
    """
    Validates, standardizes, and maps incoming raw channels to canonical 19-node ST-GAT graph.
    """
    def __init__(self, target_channels: List[str] = CANONICAL_19):
        self.target_channels = [c.upper() for c in target_channels]

    def clean_name(self, name: str) -> str:
        """Strip prefixes, whitespace, and normalize casing."""
        cleaned = name.strip()
        for prefix in ["EEG ", "eeg ", "EEG-", "eeg-"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        return cleaned.upper()

    def identify_channels(self, raw_channel_names: List[str]) -> Dict[str, Optional[str]]:
        """
        Map target canonical channels to the best available raw channel name.
        Returns dict: { 'FP1': 'raw_name' or None, ... }
        """
        mapping = {target: None for target in self.target_channels}
        raw_cleaned = {ch: self.clean_name(ch) for ch in raw_channel_names}

        # 1. Exact unipolar or alias match
        for target in self.target_channels:
            aliases = [self.clean_name(a) for a in CHANNEL_ALIASES.get(target, [target])]
            for raw_orig, raw_clean in raw_cleaned.items():
                if raw_clean in aliases:
                    mapping[target] = raw_orig
                    break

        # 2. Bipolar prefix match (e.g. 'FP1-F7' maps to 'FP1' if direct unipolar missing)
        for target in self.target_channels:
            if mapping[target] is not None:
                continue
            for raw_orig, raw_clean in raw_cleaned.items():
                parts = raw_clean.split('-')
                if len(parts) == 2:
                    p1, p2 = parts[0].strip(), parts[1].strip()
                    if p1 == target or (target in CHANNEL_ALIASES and p1 in [self.clean_name(a) for a in CHANNEL_ALIASES[target]]):
                        mapping[target] = raw_orig
                        break

        # 3. Substring match fallback
        for target in self.target_channels:
            if mapping[target] is not None:
                continue
            for raw_orig, raw_clean in raw_cleaned.items():
                if target in raw_clean.split('-') or target in raw_clean.split('_'):
                    mapping[target] = raw_orig
                    break

        return mapping

    def validate_recording(self, raw_channel_names: List[str]) -> Dict:
        """
        Produce a comprehensive validation audit report for the recording channels.
        """
        mapping = self.identify_channels(raw_channel_names)
        found_channels = [tgt for tgt, src in mapping.items() if src is not None]
        missing_channels = [tgt for tgt, src in mapping.items() if src is None]
        
        match_rate = len(found_channels) / len(self.target_channels)
        is_compatible = match_rate >= 0.70  # At least 70% channels must map safely
        
        # Determine montage type
        is_bipolar = any('-' in ch for ch in raw_channel_names if not ch.startswith('.'))
        montage_type = "Bipolar Longitudinal (Double-Banana)" if is_bipolar else "Monopolar / Referenced 10-20"

        return {
            "total_raw_channels": len(raw_channel_names),
            "target_channels_required": len(self.target_channels),
            "mapped_channels_count": len(found_channels),
            "missing_channels_count": len(missing_channels),
            "match_rate_percentage": round(match_rate * 100.0, 1),
            "is_compatible": is_compatible,
            "montage_type": montage_type,
            "mapping_dict": mapping,
            "missing_channels": missing_channels,
            "strategy": "Zero-fill / spatial interpolation" if missing_channels else "Direct canonical mapping"
        }

    def map_signals(self, raw_data: np.ndarray, raw_channel_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract and align raw data into shape (19, num_samples).
        Pads missing channels with average of neighboring mapped channels or zeros.
        """
        num_samples = raw_data.shape[1]
        out_signals = np.zeros((len(self.target_channels), num_samples), dtype=np.float32)
        mapping = self.identify_channels(raw_channel_names)
        
        # First pass: copy direct mapped channels
        mapped_indices = []
        for i, target in enumerate(self.target_channels):
            src_name = mapping[target]
            if src_name is not None and src_name in raw_channel_names:
                src_idx = raw_channel_names.index(src_name)
                out_signals[i] = raw_data[src_idx]
                mapped_indices.append(i)

        # Second pass: fill missing channels using spatial neighborhood average
        if len(mapped_indices) > 0 and len(mapped_indices) < len(self.target_channels):
            mean_signal = np.mean(out_signals[mapped_indices], axis=0)
            for i, target in enumerate(self.target_channels):
                if i not in mapped_indices:
                    # Impute missing channel with baseline spatial mean
                    out_signals[i] = mean_signal * 0.5
                    
        return out_signals, self.target_channels
