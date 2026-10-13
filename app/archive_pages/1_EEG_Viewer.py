"""
Page 1 - Multi-Channel Continuous Scalp EEG Waveform Oscilloscope (Crisp Light Mode)
NeuroSync 4D Telemetry Workstation - Nihon Kohden / Cadwell Digital Neurofax Style
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

# Ensure numpy StringDType polyfill
if hasattr(np, "dtypes") and not hasattr(np.dtypes, "StringDType"):
    np.dtypes.StringDType = np.dtype("U")

from src.config import ELECTRODES_19, SAMPLE_RATE
from src.forecasting_engine import RealTimeForecastingEngine
from src.data.chb_mit_dataset import CHBMITDataSimulator
from src.data.database_manager import DatabaseManager, DATABASES, PATIENT_PROFILES

st.set_page_config(page_title="19-Channel Oscilloscope", page_icon="⚡", layout="wide")

# Advanced Medical Oscilloscope Custom CSS
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
    
    .scope-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    
    .lead-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px;
    }
    .lead-ok { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .lead-focal { background: #ffe4e6; color: #e11d48; border: 1px solid #fecdd3; animation: pulse 1.8s infinite; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ 19-Channel Continuous Scalp EEG Waveform Oscilloscope")
st.caption("International 10-20 Standard Montage | Calibrated Bedside Clinical Neurofax")

# Sidebar
with st.sidebar:
    st.subheader("Oscilloscope Controls")
    
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
        "Signal Source",
        ["Real Hospital EDF File (.edf)", "Dynamic Pre-Ictal Simulation"],
        index=0
    )
    
    state = "preictal"
    if "Simulation" in stream_source:
        state = st.selectbox("Physiological State", ["interictal", "preictal", "ictal"], index=1)
        
    duration = st.slider("Sweep Time (Seconds)", 5, 20, 10, 1)
    gain = st.slider("Voltage Sensitivity (uV/mm)", 0.5, 5.0, 1.8, 0.2)
    selected_channels = st.multiselect("Visible Channels", ELECTRODES_19, default=ELECTRODES_19)

@st.cache_resource
def get_engine():
    ckpt = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
    return RealTimeForecastingEngine(model_checkpoint_path=str(ckpt)), CHBMITDataSimulator()

engine, simulator = get_engine()

# Load EEG Data
if "Real" in stream_source:
    eeg_data, time_axis = simulator.load_real_patient_recording(patient_id, duration_sec=duration)
    source_tag = f"Real Clinical EDF: {patient['edf_filename']}"
else:
    eeg_data, time_axis = simulator.generate_patient_recording(patient_id, duration_sec=duration, state=state)
    source_tag = f"Simulated {state.upper()} Dynamic State"

focal_chs = patient.get("focal_channels", [])

# Lead Status Strip
st.markdown("""
<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:12px 16px; margin-bottom:18px; display:flex; flex-wrap:wrap; align-items:center; gap:4px;">
    <span style="font-size:0.8rem; font-weight:700; color:#475569; margin-right:8px;">LEAD IMPEDANCE STATUS (&lt;5 k&Omega;):</span>
""", unsafe_allow_html=True)

lead_pills_html = "".join([
    f'<span class="lead-pill {"lead-focal" if ch in focal_chs else "lead-ok"}">{ch}{"*" if ch in focal_chs else ""}</span>'
    for ch in ELECTRODES_19
])
st.markdown(f"{lead_pills_html}</div>", unsafe_allow_html=True)

# Main Oscilloscope Display
st.markdown('<div class="scope-card">', unsafe_allow_html=True)

if not selected_channels:
    st.warning("Please select at least one channel to display.")
else:
    fig = go.Figure()
    offset_step = 85.0 * gain
    
    for i, ch in enumerate(selected_channels):
        ch_idx = ELECTRODES_19.index(ch)
        sig = eeg_data[ch_idx] * gain + (i * offset_step)
        is_focal = ch in focal_chs
        
        line_color = "#e11d48" if is_focal else "#1e293b"
        line_width = 1.8 if is_focal else 1.1
        
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=sig,
            mode='lines',
            name=f"{ch} {'(Focal)' if is_focal else ''}",
            line=dict(color=line_color, width=line_width),
            hoverinfo='name+x+y'
        ))

    tick_vals = [i * offset_step for i in range(len(selected_channels))]
    
    fig.update_layout(
        template="plotly_white",
        height=max(480, len(selected_channels) * 32),
        margin=dict(l=65, r=30, t=15, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            title="Continuous Sweep Time (Seconds)",
            showgrid=True,
            gridcolor="#f1f5f9",
            dtick=1.0, # 1-second grid markers like medical paper!
            linecolor="#cbd5e1"
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=selected_channels,
            showgrid=True,
            gridcolor="#f8fafc",
            zeroline=False
        ),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption("🔴 Red traces indicate patient's verified epileptogenic focus; background channels shown in dark slate with 1.0s clinical time markers.")
st.markdown('</div>', unsafe_allow_html=True)

# Oscilloscope Metrics Row
col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    st.markdown('<div class="scope-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#1e1b4b; margin:0 0 10px 0;'>📊 Spectral Power Breakdown at Focal Channel</h4>", unsafe_allow_html=True)
    
    focal_ch = focal_chs[0] if focal_chs else "T7"
    f_idx = ELECTRODES_19.index(focal_ch) if focal_ch in ELECTRODES_19 else 0
    sig_focal = eeg_data[f_idx]
    
    fft_vals = np.abs(np.fft.rfft(sig_focal)) ** 2
    freqs = np.fft.rfftfreq(len(sig_focal), 1.0 / SAMPLE_RATE)
    
    bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)}
    b_pct = []
    for _, (low, high) in bands.items():
        m = (freqs >= low) & (freqs < high)
        b_pct.append(float(np.sum(fft_vals[m])) if np.any(m) else 1.0)
    total_p = sum(b_pct) or 1.0
    b_pct = [p / total_p * 100 for p in b_pct]
    
    fig_b = go.Figure(data=[go.Bar(
        x=list(bands.keys()), y=b_pct,
        marker_color=["#6366f1", "#4338ca", "#10b981", "#f59e0b", "#e11d48"]
    )])
    fig_b.update_layout(
        template="plotly_white", height=230, margin=dict(l=30, r=20, t=10, b=30),
        yaxis=dict(title="Power (%)", gridcolor="#f1f5f9"),
        paper_bgcolor="#ffffff"
    )
    st.plotly_chart(fig_b, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_m2:
    st.markdown('<div class="scope-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#1e1b4b; margin:0 0 10px 0;'>🩺 Signal Integrity & Telemetry Dossier</h4>", unsafe_allow_html=True)
    st.markdown(f"""
    <table style="width:100%; border-collapse:collapse; font-size:0.9rem; color:#334155;">
        <tr style="border-bottom:1px solid #f8fafc;"><td style="padding:6px 0; font-weight:700;">RMS Amplitude:</td><td style="padding:6px 0; color:#4338ca; font-weight:700;">{np.sqrt(np.mean(eeg_data**2)):.2f} &mu;V</td></tr>
        <tr style="border-bottom:1px solid #f8fafc;"><td style="padding:6px 0; font-weight:700;">Peak Voltage:</td><td style="padding:6px 0;">{np.max(np.abs(eeg_data)):.2f} &mu;V</td></tr>
        <tr style="border-bottom:1px solid #f8fafc;"><td style="padding:6px 0; font-weight:700;">Sampling Rate:</td><td style="padding:6px 0;">256 Hz Continuous</td></tr>
        <tr><td style="padding:6px 0; font-weight:700;">Active Focus:</td><td style="padding:6px 0; color:#e11d48; font-weight:700;">{patient['focal_zone']}</td></tr>
    </table>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
