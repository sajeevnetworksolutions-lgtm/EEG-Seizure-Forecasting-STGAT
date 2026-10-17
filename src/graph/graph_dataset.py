# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
PyTorch Graph Sequence Dataset for ST-GAT
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class STGATGraphDataset(Dataset):
    """
    PyTorch Dataset yielding (X_sequence, Adj_sequence, y_label, metadata)
    for Spatio-Temporal Graph Attention Network training and inference.
    """
    def __init__(self,
                 sequences: np.ndarray,
                 adjacencies: np.ndarray,
                 labels: np.ndarray,
                 patient_ids: list = None,
                 timestamps: list = None):
        """
        Parameters
        ----------
        sequences : np.ndarray
            Shape (N, seq_len, num_nodes, in_features)
        adjacencies : np.ndarray
            Shape (N, seq_len, num_nodes, num_nodes)
        labels : np.ndarray
            Shape (N,) or (N, 1) binary class labels (0 = interictal, 1 = preictal)
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.adjacencies = torch.tensor(adjacencies, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        self.patient_ids = patient_ids if patient_ids is not None else ["anon"] * len(labels)
        self.timestamps = timestamps if timestamps is not None else list(range(len(labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.adjacencies[idx],
            self.labels[idx],
            self.patient_ids[idx],
            self.timestamps[idx]
        )

def create_dataloader(sequences, adjacencies, labels, batch_size=16, shuffle=True, **kwargs):
    dataset = STGATGraphDataset(sequences, adjacencies, labels, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
