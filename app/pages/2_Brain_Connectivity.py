"""
Page 2 - Dynamic Brain Connectivity Graph & Topographic Visualizer
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ELECTRODES_19, ELECTRODE_COORDS_2D, EEG_BANDS, CHB_MIT_PATIENTS
from src.forecasting_engine import RealTimeForecastingEngine
from src.data.chb_mit_dataset import CHBMITDataSimulator

st.set_page_config(page_title="Brain Connectivity", page_icon="🕸️", layout="wide")

st.title("🕸️ Dynamic Brain Functional Connectivity (Phase Locking Value)")
st.caption("Time-Resolved Inter-Regional Synchrony & Scalp 10-20 Dynamic Graph Topography")

with st.sidebar:
    st.subheader("Connectivity Options")
    patient = st.selectbox("Patient", CHB_MIT_PATIENTS, index=0)
    state = st.selectbox("State", ["interictal", "preictal", "ictal"], index=1)
    plv_thresh = st.slider("PLV Synchrony Edge Threshold", 0.1, 0.8, 0.35, 0.05)

@st.cache_resource
def get_system():
    ckpt = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
    return RealTimeForecastingEngine(model_checkpoint_path=str(ckpt)), CHBMITDataSimulator()

engine, simulator = get_system()
raw_eeg, _ = simulator.generate_patient_recording(patient, duration_sec=8, state=state)
engine.ingest_sample_block(raw_eeg)
res = engine.process_latest_window()

tab1, tab2, tab3 = st.tabs(["2D Scalp Topography", "PLV Synchrony Matrix", "Frequency Band Power"])

with tab1:
    fig_topo = go.Figure()
    
    # Head outline circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig_topo.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color='#475569', width=2),
        hoverinfo='none', showlegend=False
    ))
    # Nose pointer
    fig_topo.add_trace(go.Scatter(
        x=[-0.1, 0.0, 0.1], y=[1.0, 1.15, 1.0],
        mode='lines', line=dict(color='#475569', width=2),
        hoverinfo='none', showlegend=False
    ))
    
    # PLV Edges
    plv = res["plv_matrix"]
    for i in range(len(ELECTRODES_19)):
        for j in range(i+1, len(ELECTRODES_19)):
            sync_val = plv[i, j]
            if sync_val >= plv_thresh:
                ch1 = ELECTRODES_19[i]
                ch2 = ELECTRODES_19[j]
                x1, y1 = ELECTRODE_COORDS_2D[ch1]
                x2, y2 = ELECTRODE_COORDS_2D[ch2]
                
                fig_topo.add_trace(go.Scatter(
                    x=[x1, x2], y=[y1, y2],
                    mode='lines',
                    line=dict(
                        color=f'rgba(239, 68, 68, {min(1.0, (sync_val - plv_thresh)*2.5)})',
                        width=1.0 + sync_val * 4.0
                    ),
                    hoverinfo='text',
                    text=f"Sync {ch1} ↔ {ch2}: PLV={sync_val:.2f}",
                    showlegend=False
                ))
                
    # Nodes
    node_x = [ELECTRODE_COORDS_2D[ch][0] for ch in ELECTRODES_19]
    node_y = [ELECTRODE_COORDS_2D[ch][1] for ch in ELECTRODES_19]
    node_sizes = [14 + 18 * res["node_importance"][i] for i in range(len(ELECTRODES_19))]
    
    fig_topo.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=res["node_importance"],
            colorscale="Viridis",
            colorbar=dict(title="GAT Attention", len=0.7),
            showscale=True,
            line=dict(width=1.5, color="#f8fafc")
        ),
        text=ELECTRODES_19,
        textposition="top center",
        textfont=dict(color="#f8fafc", size=10),
        hoverinfo='text',
        hovertext=[f"Electrode: {ch}<br>GAT Attention: {res['node_importance'][i]:.3f}" for i, ch in enumerate(ELECTRODES_19)],
        showlegend=False
    ))
    
    fig_topo.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0b132b",
        paper_bgcolor="#1e293b",
        xaxis=dict(visible=False, range=[-1.25, 1.25]),
        yaxis=dict(visible=False, range=[-1.25, 1.25])
    )
    st.plotly_chart(fig_topo, use_container_width=True)

with tab2:
    fig_mat = px.imshow(
        res["plv_matrix"],
        x=ELECTRODES_19,
        y=ELECTRODES_19,
        color_continuous_scale="RdYlBu_r",
        range_color=[0.0, 1.0],
        labels=dict(x="Channel", y="Channel", color="PLV")
    )
    fig_mat.update_layout(
        height=480,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0b132b",
        font=dict(color="#94a3b8")
    )
    st.plotly_chart(fig_mat, use_container_width=True)

with tab3:
    band_names = list(EEG_BANDS.keys())
    mean_band_pwr = np.mean(res["band_power"], axis=0)
    fig_band = px.bar(
        x=band_names,
        y=mean_band_pwr * 100,
        color=band_names,
        color_discrete_sequence=["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#ef4444"],
        labels=dict(x="Neurological Frequency Band", y="Relative Power (%)")
    )
    fig_band.update_layout(
        height=450,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0b132b",
        font=dict(color="#94a3b8"),
        showlegend=False
    )
    st.plotly_chart(fig_band, use_container_width=True)
