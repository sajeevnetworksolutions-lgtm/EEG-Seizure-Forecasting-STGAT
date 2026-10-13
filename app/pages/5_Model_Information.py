"""
Page 5 - Model Architecture and Mathematical Formulations (Crisp Light Mode)
NeuroSync 4D Telemetry Workstation - ST-GAT Specifications
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Model Architecture & Math", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(#e2e8f0 1.2px, transparent 1.2px);
        background-size: 24px 24px;
        color: #0f172a;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    .math-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 ST-GAT Architecture & Mathematical Specifications")
st.caption("Theoretical & Algorithmic Blueprint for M.Tech Dissertation Viva Defense")

# 1. Mathematical Formulations Card
st.markdown('<div class="math-card">', unsafe_allow_html=True)
st.markdown("<h3 style='color:#1e1b4b; margin:0 0 14px 0;'>1. Core Mathematical Formulations</h3>", unsafe_allow_html=True)

st.markdown("#### A. Dynamic Phase Locking Value (PLV)")
st.latex(r"z_i(t) = x_i(t) + j \mathcal{H}[x_i(t)] = A_i(t) e^{j \phi_i(t)}")
st.latex(r"\text{PLV}_{ij} = \frac{1}{T} \left| \sum_{t=1}^T \exp\left(j (\phi_i(t) - \phi_j(t))\right) \right| \in [0, 1]")

st.markdown("#### B. Dynamic Spatial Graph Attention Coefficient")
st.latex(r"\alpha_{ij}^t = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^t \,\|\, \mathbf{W} h_j^t]\right)\right) \cdot \mathcal{A}_{ij}^t}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^t \,\|\, \mathbf{W} h_k^t]\right)\right) \cdot \mathcal{A}_{ik}^t}")

st.markdown("#### C. Temporal Gated Recurrent Unit (GRU) State Propagation")
st.latex(r"H^{t+1} = \text{GRU}\left(\text{SpatialGAT}(H^t, \mathcal{A}^t),\, H^{t-1}\right)")

st.markdown("#### D. Class-Imbalanced Focal Loss Optimization Objective")
st.latex(r"\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad (\gamma = 2.0, \, \alpha = 0.75)")
st.markdown('</div>', unsafe_allow_html=True)

# 2. Architectural Parameter Matrix Card
st.markdown('<div class="math-card">', unsafe_allow_html=True)
st.markdown("<h3 style='color:#1e1b4b; margin:0 0 14px 0;'>2. Layer-by-Layer Architectural Parameter Matrix</h3>", unsafe_allow_html=True)

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
    "Input Tensor Shape": [
        "(B, T, 19, 256)",
        "(B, T, 19, 64)",
        "(B, T, 19, 64)",
        "(B, T, 19, 64)",
        "(B, T, 128)",
        "(B, 128)"
    ],
    "Output Tensor Shape": [
        "(B, T, 19, 64)",
        "(B, T, 19, 64)",
        "(B, T, 19, 64)",
        "(B, T, 128)",
        "(B, 128)",
        "(B, 1) in [0, 1]"
    ],
    "Trainable Parameters": [
        "8,448",
        "16,640",
        "16,640",
        "0 (Functional Pool)",
        "98,816",
        "8,321"
    ]
})

st.dataframe(df_layers, hide_index=True, use_container_width=True)
st.caption("Total Trainable Model Parameters: 148,865 &bull; Hardware: Optimized for Real-Time Clinical Inference (3.8 ms / epoch).")
st.markdown('</div>', unsafe_allow_html=True)
