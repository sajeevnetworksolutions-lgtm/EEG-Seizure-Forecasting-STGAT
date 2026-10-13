"""
Page 2 - Dynamic Brain Connectivity Graph & Topographic Visualizer (Crisp Light Mode)
NeuroSync 4D Telemetry Workstation - Spatial-Temporal Graph Attention Dynamics
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Windows PyTorch DLL initialization fix
if sys.platform == "win32":
    try:
        import site
        for sp in site.getsitepackages():
            torch_lib = Path(sp) / "torch" / "lib"
            if torch_lib.exists():
                os.add_dll_directory(str(torch_lib))
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure numpy StringDType polyfill
if hasattr(np, "dtypes") and not hasattr(np.dtypes, "StringDType"):
    np.dtypes.StringDType = np.dtype("U")

from src.config import ELECTRODES_19, ELECTRODE_COORDS_2D
from src.forecasting_engine import RealTimeForecastingEngine
from src.data.chb_mit_dataset import CHBMITDataSimulator
from src.data.database_manager import DatabaseManager, DATABASES, PATIENT_PROFILES

st.set_page_config(page_title="Brain Connectivity", page_icon="🕸️", layout="wide")

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
    
    .network-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
</style>
""", unsafe_allow_html=True)

st.title("🕸️ Dynamic Brain Functional Connectivity & Topography")
st.caption("Time-Resolved Phase Locking Value (PLV) & Scalp 10-20 Graph Attention Mapping")

# Sidebar
with st.sidebar:
    st.subheader("Graph Hyperparameters")
    
    db_key = st.selectbox(
        "Clinical Database",
        options=list(DATABASES.keys()),
        format_func=lambda k: DATABASES[k]["name"],
        index=0
    )
    
    avail_patients = [p for p in PATIENT_PROFILES.keys() if PATIENT_PROFILES[p]["database"] == db_key]
    if not avail_patients:
        avail_patients = list(PATIENT_PROFILES.keys())
        
    patient_id = st.selectbox(
        "Patient Subject",
        options=avail_patients,
        format_func=lambda p: PATIENT_PROFILES[p]["display_name"],
        index=0
    )
    patient = PATIENT_PROFILES[patient_id]
    
    stream_source = st.radio(
        "Signal Stream",
        ["Real Hospital EDF File (.edf)", "Dynamic Pre-Ictal Simulation"],
        index=0
    )
    
    state = "preictal"
    if "Simulation" in stream_source:
        state = st.selectbox("Simulation Phase", ["interictal", "preictal", "ictal"], index=1)
        
    plv_thresh = st.slider("PLV Synchrony Threshold", 0.10, 0.80, 0.35, 0.05)

@st.cache_resource
def get_system():
    ckpt = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
    return RealTimeForecastingEngine(model_checkpoint_path=str(ckpt)), CHBMITDataSimulator()

engine, simulator = get_system()

# Load Data
if "Real" in stream_source:
    raw_eeg, _ = simulator.load_real_patient_recording(patient_id, duration_sec=8)
    source_tag = f"Real Clinical EDF: {patient['edf_filename']}"
else:
    raw_eeg, _ = simulator.generate_patient_recording(patient_id, duration_sec=8, state=state)
    source_tag = f"Simulated {state.upper()} Dynamic Stream"

engine.ingest_sample_block(raw_eeg)
res = engine.process_latest_window()

plv_mat = res["plv_matrix"]
attn_weights = res["gat_attention"]
node_imp = res["node_importance"]
focal_chs = patient.get("focal_channels", [])

col_topo, col_matrix = st.columns([1.1, 0.9])

with col_topo:
    st.markdown('<div class="network-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#1e1b4b; margin:0 0 10px 0;'>🧠 2D Scalp 10-20 Functional Topography</h4>", unsafe_allow_html=True)
    
    fig_topo = go.Figure()
    
    # Outer head contour in light clinical theme
    theta = np.linspace(0, 2 * np.pi, 200)
    fig_topo.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', line=dict(color='#cbd5e1', width=2.5), hoverinfo='none', showlegend=False))
    fig_topo.add_trace(go.Scatter(x=[-0.12, 0.0, 0.12], y=[0.98, 1.15, 0.98], mode='lines', line=dict(color='#cbd5e1', width=2.5), hoverinfo='none', showlegend=False))
    fig_topo.add_trace(go.Scatter(x=[-1.0, -1.08, -1.08, -1.0], y=[0.12, 0.08, -0.08, -0.12], mode='lines', line=dict(color='#cbd5e1', width=2), hoverinfo='none', showlegend=False))
    fig_topo.add_trace(go.Scatter(x=[1.0, 1.08, 1.08, 1.0], y=[0.12, 0.08, -0.08, -0.12], mode='lines', line=dict(color='#cbd5e1', width=2), hoverinfo='none', showlegend=False))

    # Add PLV Edges
    for i in range(19):
        for j in range(i + 1, 19):
            val = plv_mat[i, j]
            if val >= plv_thresh:
                ch_i, ch_j = ELECTRODES_19[i], ELECTRODES_19[j]
                xi, yi = ELECTRODE_COORDS_2D[ch_i]
                xj, yj = ELECTRODE_COORDS_2D[ch_j]
                
                is_focal_edge = (ch_i in focal_chs) or (ch_j in focal_chs)
                edge_color = f"rgba(225, 29, 72, {min(1.0, val)})" if is_focal_edge else f"rgba(67, 56, 202, {min(0.8, val)})"
                
                fig_topo.add_trace(go.Scatter(
                    x=[xi, xj], y=[yi, yj], mode='lines',
                    line=dict(color=edge_color, width=val * 4.5),
                    hoverinfo='text', text=f"PLV: {ch_i} - {ch_j} = {val:.3f}",
                    showlegend=False
                ))

    # Scalp Electrodes (Nodes)
    node_x = [ELECTRODE_COORDS_2D[ch][0] for ch in ELECTRODES_19]
    node_y = [ELECTRODE_COORDS_2D[ch][1] for ch in ELECTRODES_19]
    node_colors = ["#e11d48" if ch in focal_chs else "#4338ca" for ch in ELECTRODES_19]
    node_sizes = [24 + imp * 16 for imp in node_imp]

    fig_topo.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors, line=dict(color='#ffffff', width=2)),
        text=ELECTRODES_19, textposition="top center",
        textfont=dict(size=11, color='#0f172a', family="Plus Jakarta Sans, sans-serif"),
        hoverinfo='text',
        hovertext=[f"Electrode: {ch}<br>GAT Attention: {imp*100:.1f}%<br>{'🚨 Focal Epileptogenic Node' if ch in focal_chs else 'Normal Node'}" for ch, imp in zip(ELECTRODES_19, node_imp)],
        showlegend=False
    ))

    fig_topo.update_layout(
        template="plotly_white", height=450, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.25, 1.25]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.25, 1.25]),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff"
    )
    st.plotly_chart(fig_topo, use_container_width=True)
    st.caption("🔴 Red nodes highlight patient's primary focus; 🟣 Deep Indigo denotes resting background brain circuitry.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_matrix:
    st.markdown('<div class="network-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#1e1b4b; margin:0 0 10px 0;'>📊 19×19 Instantaneous PLV Heatmap</h4>", unsafe_allow_html=True)
    
    fig_mat = px.imshow(
        plv_mat, x=ELECTRODES_19, y=ELECTRODES_19,
        color_continuous_scale="Purples", range_color=[0.0, 1.0]
    )
    fig_mat.update_layout(
        template="plotly_white", height=450, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff"
    )
    st.plotly_chart(fig_mat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
