"""
Page 1 - Multi-Channel Continuous Scalp EEG Waveform Viewer (Crisp Light Mode)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
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

from src.config import ELECTRODES_19, CHB_MIT_PATIENTS, SAMPLE_RATE
from src.forecasting_engine import RealTimeForecastingEngine
from src.data.chb_mit_dataset import CHBMITDataSimulator

st.set_page_config(page_title="19-Channel EEG Viewer", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #0f172a; }
    section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

st.title("📈 19-Channel Continuous Scalp EEG Waveform Viewer")
st.caption("Standard 10-20 International Montage | Real-Time Filtered Signal Stream")

# Sidebar settings
with st.sidebar:
    st.subheader("Signal Controls")
    selected_patient = st.selectbox("Patient", CHB_MIT_PATIENTS, index=0)
    state = st.selectbox("State", ["interictal", "preictal", "ictal"], index=1)
    gain = st.slider("Signal Gain (uV)", 1.0, 5.0, 2.0, 0.5)
    selected_channels = st.multiselect("Visible Channels", ELECTRODES_19, default=ELECTRODES_19)

@st.cache_resource
def get_engine():
    ckpt = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
    return RealTimeForecastingEngine(model_checkpoint_path=str(ckpt)), CHBMITDataSimulator()

engine, simulator = get_engine()

# Generate EEG
raw_eeg, t_ax = simulator.generate_patient_recording(selected_patient, duration_sec=8, state=state)
engine.ingest_sample_block(raw_eeg)
res = engine.process_latest_window()

eeg_traces = res["recent_eeg"] # (19, samples)
n_samples = eeg_traces.shape[1]
t_sec = np.linspace(0, 4.0, n_samples)

# Plot Stacked Traces on Clean White Canvas
fig = go.Figure()
offset_spacing = 3.0 * gain

channels_to_plot = [ch for ch in ELECTRODES_19 if ch in selected_channels]

# Distinct, high-contrast trace palette for light mode
trace_colors = [
    "#0284c7", "#2563eb", "#4f46e5", "#7c3aed", "#9333ea",
    "#c026d3", "#db2777", "#e11d48", "#ea580c", "#d97706",
    "#059669", "#0d9488", "#0891b2", "#0284c7", "#2563eb",
    "#4f46e5", "#7c3aed", "#db2777", "#ea580c"
]

for i, ch_name in enumerate(channels_to_plot):
    idx = ELECTRODES_19.index(ch_name)
    offset = (len(channels_to_plot) - 1 - i) * offset_spacing
    sig = eeg_traces[idx] * gain + offset
    color = trace_colors[idx % len(trace_colors)]
    
    fig.add_trace(go.Scatter(
        x=t_sec, y=sig,
        mode="lines",
        line=dict(color=color, width=1.5),
        name=ch_name,
        hoverinfo="name+y"
    ))

fig.update_layout(
    height=640,
    margin=dict(l=60, r=20, t=20, b=30),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(color="#1e293b", family="sans-serif"),
    xaxis=dict(
        title="Time Window (Seconds)",
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1"
    ),
    yaxis=dict(
        tickvals=[(len(channels_to_plot) - 1 - i) * offset_spacing for i in range(len(channels_to_plot))],
        ticktext=channels_to_plot,
        showgrid=True,
        gridcolor="#f8fafc",
        linecolor="#cbd5e1"
    ),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
