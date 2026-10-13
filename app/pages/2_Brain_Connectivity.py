"""
Page 2 - Dynamic Brain Connectivity Graph & Topographic Visualizer (Crisp Light Mode)
Multi-Database & Multi-Patient Phase Locking Value (PLV) Network Analysis
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

from src.config import ELECTRODES_19, ELECTRODE_COORDS_2D, EEG_BANDS
from src.forecasting_engine import RealTimeForecastingEngine
from src.data.chb_mit_dataset import CHBMITDataSimulator
from src.data.database_manager import DatabaseManager, DATABASES, PATIENT_PROFILES

st.set_page_config(page_title="Brain Connectivity", page_icon="🕸️", layout="wide")

# Enforce Pure Crisp Light Mode
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #0f172a; }
    section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
    .header-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)

st.title("🕸️ Dynamic Brain Functional Connectivity (Phase Locking Value)")
st.caption("Time-Resolved Inter-Regional Synchrony & Scalp 10-20 Dynamic Graph Topography")

# Sidebar
with st.sidebar:
    st.subheader("Connectivity Options")
    
    # Database
    db_key = st.selectbox(
        "Clinical Database",
        options=list(DATABASES.keys()),
        format_func=lambda k: DATABASES[k]["name"],
        index=0
    )
    
    # Patient
    avail_patients = [p for p in PATIENT_PROFILES.keys() if PATIENT_PROFILES[p]["database"] == db_key]
    if not avail_patients:
        avail_patients = list(PATIENT_PROFILES.keys())
        
    patient_id = st.selectbox(
        "Patient Cohort",
        options=avail_patients,
        format_func=lambda p: PATIENT_PROFILES[p]["display_name"],
        index=0
    )
    patient = PATIENT_PROFILES[patient_id]
    
    stream_source = st.radio(
        "Signal Source",
        ["Real Hospital EDF File (.edf)", "Dynamic Pre-Ictal Simulation"],
        index=0
    )
    
    state = "preictal"
    if "Simulation" in stream_source:
        state = st.selectbox("Physiological State", ["interictal", "preictal", "ictal"], index=1)
        
    plv_thresh = st.slider("PLV Synchrony Edge Threshold", 0.10, 0.80, 0.35, 0.05)

@st.cache_resource
def get_system():
    ckpt = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
    return RealTimeForecastingEngine(model_checkpoint_path=str(ckpt)), CHBMITDataSimulator()

engine, simulator = get_system()

# Load EEG Data
if "Real" in stream_source:
    raw_eeg, _ = simulator.load_real_patient_recording(patient_id, duration_sec=8)
    source_tag = f"Real Clinical EDF: {patient['edf_filename']}"
else:
    raw_eeg, _ = simulator.generate_patient_recording(patient_id, duration_sec=8, state=state)
    source_tag = f"Simulated {state.upper()} Dynamic State"

engine.ingest_sample_block(raw_eeg)
res = engine.process_latest_window()

plv_mat = res["plv_matrix"]
attn_weights = res["gat_attention"]
node_imp = res["node_importance"]
focal_chs = patient.get("focal_channels", [])

# Header Card
st.markdown(f"""
<div class="header-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h4 style="margin:0; color:#0284c7; font-weight:700;">{patient['display_name']}</h4>
            <span style="color:#64748b; font-size:0.9rem;">
                Diagnosis: <b>{patient['diagnosis']}</b> &bull; Epileptogenic Zone: <b style="color:#dc2626;">{patient['focal_zone']}</b>
            </span>
        </div>
        <span style="background-color:#e0f2fe; color:#0369a1; padding:4px 12px; border-radius:6px; font-weight:700; font-size:0.85rem; border:1px solid #bae6fd;">
            {source_tag}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

col_topo, col_matrix = st.columns([1.1, 0.9])

with col_topo:
    st.subheader("2D Scalp 10-20 Topographic Brain Graph")
    
    fig_topo = go.Figure()
    
    # Outer head contour in light clinical theme
    theta = np.linspace(0, 2 * np.pi, 200)
    fig_topo.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color='#cbd5e1', width=2.5),
        hoverinfo='none', showlegend=False
    ))
    
    # Nose indicator at top
    fig_topo.add_trace(go.Scatter(
        x=[-0.12, 0.0, 0.12], y=[0.98, 1.15, 0.98],
        mode='lines', line=dict(color='#cbd5e1', width=2.5),
        hoverinfo='none', showlegend=False
    ))
    
    # Left and Right ears
    fig_topo.add_trace(go.Scatter(
        x=[-1.0, -1.08, -1.08, -1.0], y=[0.12, 0.08, -0.08, -0.12],
        mode='lines', line=dict(color='#cbd5e1', width=2),
        hoverinfo='none', showlegend=False
    ))
    fig_topo.add_trace(go.Scatter(
        x=[1.0, 1.08, 1.08, 1.0], y=[0.12, 0.08, -0.08, -0.12],
        mode='lines', line=dict(color='#cbd5e1', width=2),
        hoverinfo='none', showlegend=False
    ))

    # Add PLV Functional Connectivity Edges
    for i in range(19):
        for j in range(i + 1, 19):
            val = plv_mat[i, j]
            if val >= plv_thresh:
                ch_i = ELECTRODES_19[i]
                ch_j = ELECTRODES_19[j]
                xi, yi = ELECTRODE_COORDS_2D[ch_i]
                xj, yj = ELECTRODE_COORDS_2D[ch_j]
                
                is_focal_edge = (ch_i in focal_chs) or (ch_j in focal_chs)
                edge_color = f"rgba(220, 38, 38, {min(1.0, val)})" if is_focal_edge else f"rgba(2, 132, 199, {min(0.8, val)})"
                
                fig_topo.add_trace(go.Scatter(
                    x=[xi, xj], y=[yi, yj],
                    mode='lines',
                    line=dict(color=edge_color, width=val * 4.5),
                    hoverinfo='text',
                    text=f"PLV Synchrony: {ch_i} - {ch_j} = {val:.3f}",
                    showlegend=False
                ))

    # Add Scalp Electrodes (Nodes)
    node_x = [ELECTRODE_COORDS_2D[ch][0] for ch in ELECTRODES_19]
    node_y = [ELECTRODE_COORDS_2D[ch][1] for ch in ELECTRODES_19]

    
    node_colors = []
    node_sizes = []
    for ch, imp in zip(ELECTRODES_19, node_imp):
        if ch in focal_chs:
            node_colors.append("#dc2626") # red focal node
            node_sizes.append(26 + imp * 14)
        else:
            node_colors.append("#0284c7") # clinical blue
            node_sizes.append(18 + imp * 10)

    fig_topo.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors, line=dict(color='#ffffff', width=2)),
        text=ELECTRODES_19,
        textposition="top center",
        textfont=dict(size=11, color='#0f172a', family="Inter, sans-serif"),
        hoverinfo='text',
        hovertext=[f"Electrode: {ch}<br>GAT Attention: {imp*100:.1f}%<br>{'🚨 Focal Epileptogenic Node' if ch in focal_chs else 'Normal Node'}" for ch, imp in zip(ELECTRODES_19, node_imp)],
        showlegend=False
    ))

    fig_topo.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.25, 1.25]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.25, 1.25]),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff"
    )
    st.plotly_chart(fig_topo, use_container_width=True)
    st.caption("🔴 Red nodes & edges highlight patient's primary epileptogenic focus; 🔵 Blue indicates baseline neural network synchrony.")

with col_matrix:
    st.subheader("19×19 Instantaneous PLV Matrix")
    
    fig_mat = px.imshow(
        plv_mat,
        labels=dict(x="Electrode", y="Electrode", color="PLV Synchrony"),
        x=ELECTRODES_19,
        y=ELECTRODES_19,
        color_continuous_scale="Blues",
        range_color=[0.0, 1.0]
    )
    fig_mat.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a")
    )
    st.plotly_chart(fig_mat, use_container_width=True)

# Band Power Breakdown Across Electrodes
st.divider()
st.subheader("Multi-Frequency Band Power Distribution across 19 Channels")

band_pwr = res["band_power"]
fig_bp = go.Figure()

band_colors = {
    "Delta (0.5-4Hz)": "#3b82f6",
    "Theta (4-8Hz)": "#0284c7",
    "Alpha (8-13Hz)": "#10b981",
    "Beta (13-30Hz)": "#f59e0b",
    "Gamma (30-45Hz)": "#ef4444"
}

for b_idx, (b_name, col) in enumerate(band_colors.items()):
    fig_bp.add_trace(go.Bar(
        name=b_name,
        x=ELECTRODES_19,
        y=band_pwr[:, b_idx],
        marker_color=col
    ))

fig_bp.update_layout(
    template="plotly_white",
    barmode='stack',
    height=320,
    margin=dict(l=40, r=20, t=20, b=40),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    xaxis=dict(title="Scalp Electrode Channel (10-20 Standard)", gridcolor="#e2e8f0"),
    yaxis=dict(title="Relative Power Spectral Density", gridcolor="#e2e8f0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_bp, use_container_width=True)
