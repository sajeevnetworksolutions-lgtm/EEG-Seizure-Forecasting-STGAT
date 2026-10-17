# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Data loading and preprocessing utilities
"""

# Lazy / modular exports
def get_dataset_classes():
    from src.data.chb_mit_dataset import CHBMITDataset, CHBMITDataSimulator, load_or_generate_dataset
    return CHBMITDataset, CHBMITDataSimulator, load_or_generate_dataset
