# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Page 3 - Real-Time Seizure Risk Forecasting & Clinical Alarm Console (Crisp Light Mode)
NeuroSync 4D Telemetry Workstation - Dual-Threshold Hysteresis Early Warning Horizon
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import datetime

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

from src.forecasting_engine import RealTimeForecastingEngine
from src.data.chb_mit_dataset import CHBMITDataSimulator
from src.data.database_manager import DatabaseManager, DATABASES, PATIENT_PROFILES

st.set_page_config(page_title="Seizure Forecasting Radar", page_icon="🚨", layout="wide")

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
    
    .forecast-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
</style>
""", unsafe_allow_html=True)

st.title("🚨 Real-Time Seizure Risk Radar & Hysteresis Controller")
st.caption("Continuous Pre-Ictal Early Warning Horizon (15 - 30 Minutes Window) with Hysteresis Suppression")

# Sidebar
with st.sidebar:
    st.subheader("Radar Hyperparameters")
    
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
        "Telemetry Stream",
        ["Real Hospital EDF File (.edf)", "Dynamic Pre-Ictal Simulation"],
        index=0
    )
    
    state = "preictal"
    if "Simulation" in stream_source:
        state = st.selectbox("Simulation Phase", ["interictal", "preictal", "ictal"], index=1)
        
    alarm_threshold = st.slider("Trigger Threshold", 0.40, 0.90, 0.65, 0.05)
    clear_threshold = st.slider("Reset Threshold", 0.20, 0.60, 0.40, 0.05)
    
    if st.button("Clear Telemetry History", use_container_width=True):
        st.session_state.forecast_history = []
        st.session_state.alarm_log = []
        st.rerun()

if "forecast_history" not in st.session_state:
    st.session_state.forecast_history = []
if "alarm_log" not in st.session_state:
    st.session_state.alarm_log = []

@st.cache_resource
def get_engine(a_thresh, c_thresh):
    ckpt = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
    eng = RealTimeForecastingEngine(model_checkpoint_path=str(ckpt), alarm_threshold=a_thresh, clear_threshold=c_thresh)
    sim = CHBMITDataSimulator()
    return eng, sim

engine, simulator = get_engine(alarm_threshold, clear_threshold)

# Ingest Data
if "Real" in stream_source:
    raw_eeg, _ = simulator.load_real_patient_recording(patient_id, duration_sec=12)
else:
    raw_eeg, _ = simulator.generate_patient_recording(patient_id, duration_sec=12, state=state)

engine.ingest_sample_block(raw_eeg)
res = engine.process_latest_window()

# Append to history
st.session_state.forecast_history.append({
    "time_step": len(st.session_state.forecast_history) + 1,
    "smoothed_risk": res["smoothed_risk"],
    "alarm_active": res["alarm_active"]
})

if len(st.session_state.forecast_history) > 40:
    st.session_state.forecast_history = st.session_state.forecast_history[-40:]

if res["alarm_active"]:
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    if not st.session_state.alarm_log or st.session_state.alarm_log[-1]["status"] != "ACTIVE":
        lead_val = res["lead_time_min"] if res["lead_time_min"] else patient["metrics"]["mean_lead_time_min"]
        st.session_state.alarm_log.append({
            "timestamp": now_str,
            "patient": patient["display_name"],
            "risk_score": f"{res['smoothed_risk']*100:.1f}%",
            "lead_time": f"{lead_val:.1f} min",
            "status": "ACTIVE",
            "action": f"Administer {patient['medications'][0]} & initiate fall-safety positioning"
        })

# Main Radar Trajectory Chart
st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
st.markdown("<h4 style='color:#1e1b4b; margin:0 0 14px 0;'>📈 Continuous Seizure Probability Trajectory & Hysteresis Bounds</h4>", unsafe_allow_html=True)

df_hist = pd.DataFrame(st.session_state.forecast_history)
fig = go.Figure()

fig.add_hline(
    y=alarm_threshold, line_dash="dash", line_color="#e11d48",
    annotation_text=f"Trigger Threshold ({alarm_threshold:.2f})",
    annotation_position="top right", annotation_font_color="#e11d48"
)
fig.add_hline(
    y=clear_threshold, line_dash="dash", line_color="#10b981",
    annotation_text=f"Reset Threshold ({clear_threshold:.2f})",
    annotation_position="bottom right", annotation_font_color="#10b981"
)

fig.add_trace(go.Scatter(
    x=df_hist["time_step"],
    y=df_hist["smoothed_risk"],
    mode='lines+markers',
    name="Smoothed Risk",
    line=dict(color="#4338ca", width=3.5),
    marker=dict(size=7, color="#312e81")
))

fig.update_layout(
    template="plotly_white",
    height=360,
    margin=dict(l=40, r=40, t=20, b=40),
    yaxis=dict(title="Pre-Ictal Seizure Probability", range=[0.0, 1.05], gridcolor="#f1f5f9"),
    xaxis=dict(title="Continuous Epoch Sequence (10s Windows)", gridcolor="#f1f5f9"),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff"
)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Emergency Clinical Protocol
if res["alarm_active"]:
    lead_disp = f"{res['lead_time_min']:.1f} Minutes" if res['lead_time_min'] else f"{patient['metrics']['mean_lead_time_min']:.1f} Minutes"
    st.markdown(f"""
    <div style="background:#fff1f2; border:1px solid #fecdd3; border-left:6px solid #e11d48; border-radius:12px; padding:20px; margin-bottom:20px;">
        <h3 style="color:#be123c; margin-top:0; font-size:1.25rem;">
            🚨 PRE-ICTAL EMERGENCY CLINICAL PROTOCOL TRIGGERED
        </h3>
        <p style="color:#334155; margin-bottom:8px;">
            Sustained PLV synchrony across <b>{patient['focal_zone']}</b> crossed <b>{alarm_threshold:.2f}</b>.
            Estimated clinical onset in <b>{lead_disp}</b>.
        </p>
        <ol style="color:#0f172a; line-height:1.7; margin-bottom:0; font-weight:600;">
            <li>Assist patient into recumbent side-lying posture (prevent fall injuries).</li>
            <li>Administer prescribed fast-acting rescue dose: <i>{patient['medications'][0]}</i>.</li>
            <li>Alert bedside neurology team and ensure clear airway.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Audit Log Table
st.subheader("📋 Hospital Alarm Event Audit Log")
if st.session_state.alarm_log:
    df_log = pd.DataFrame(st.session_state.alarm_log)
    st.dataframe(df_log, hide_index=True, use_container_width=True)
    csv_data = df_log.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Clinical Alarm Audit Report (CSV)",
        data=csv_data,
        file_name=f"alarm_log_{patient['patient_id']}.csv",
        mime="text/csv"
    )
else:
    st.info("No clinical alarms triggered in the current window. Baseline neural stability confirmed.")
