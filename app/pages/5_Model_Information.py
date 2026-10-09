"""
Page 5 - Model Architecture and Mathematical Formulations
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Model Architecture", page_icon="🧬", layout="wide")

st.title("🧬 Model Architecture & Mathematical Formulations")
st.caption("M.Tech AI/ML Master Dissertation Technical Specifications")

st.subheader("1. Mathematical Formulations")

st.markdown("#### A. Dynamic Phase Locking Value (PLV)")
st.latex(r"z_i(t) = x_i(t) + j \mathcal{H}[x_i(t)] = A_i(t) e^{j \phi_i(t)}")
st.latex(r"\text{PLV}_{ij} = \frac{1}{T} \left| \sum_{t=1}^T \exp\left(j (\phi_i(t) - \phi_j(t))\right) \right| \in [0, 1]")

st.markdown("#### B. Dynamic Spatial Graph Attention Coefficient")
st.latex(r"\alpha_{ij}^t = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^t \,\|\, \mathbf{W} h_j^t]\right)\right) \cdot \mathcal{A}_{ij}^t}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^t \,\|\, \mathbf{W} h_k^t]\right)\right) \cdot \mathcal{A}_{ik}^t}")

st.markdown("#### C. Temporal Gated Recurrent Unit (GRU) State Propagation")
st.latex(r"H^{t+1} = \text{GRU}\left(\text{SpatialGAT}(H^t, \mathcal{A}^t),\, H^{t-1}\right)")

st.markdown("#### D. Class-Imbalanced Focal Loss Optimization Objective")
st.latex(r"\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad (\gamma = 2.0, \, \alpha = 0.75)")

st.divider()
st.subheader("2. Layer-by-Layer Architectural Parameter Matrix")

df_layers = pd.DataFrame({
    "Component / Layer": [
        "Temporal Wavelet Filter",
        "Spatial GAT Layer 1",
        "Spatial GAT Layer 2",
        "Spatial Readout & Pooling",
        "Temporal GRU Sequence Cell",
        "Risk Classifier Head"
    ],
    "Operation & Mechanism": [
        "1D Multi-Scale Convolutions (Delta, Theta, Alpha, Beta, Gamma)",
        "Multi-Head Dynamic Graph Attention (4 Heads)",
        "Multi-Head Dynamic Graph Attention (4 Heads)",
        "Global Mean + Max Spatial Node Pooling",
        "Recurrent Sequence Aggregation (10-window sequence)",
        "Linear + LayerNorm + Sigmoid (Pre-Ictal Risk)"
    ],
    "Input Shape": [
        "(B, 19, 256)",
        "(B, 19, 64)",
        "(B, 19, 128)",
        "(B, 19, 128)",
        "(B, 10, 128)",
        "(B, 128)"
    ],
    "Output Shape": [
        "(B, 19, 64)",
        "(B, 19, 128)",
        "(B, 19, 128)",
        "(B, 128)",
        "(B, 128)",
        "(B, 1)"
    ]
})

st.dataframe(df_layers, use_container_width=True, hide_index=True)
