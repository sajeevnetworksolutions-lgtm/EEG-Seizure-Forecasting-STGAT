"""
Page 3 - Real-Time Seizure Risk Forecasting & Clinical Alarm Engine (Crisp Light Mode)
Multi-Database & Multi-Patient Hysteresis Early Warning Horizon Controller
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

st.set_page_config(page_title="Seizure Forecasting", page_icon="🚨", layout="wide")

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
    .protocol-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #dc2626;
        border-radius: 10px;
        padding: 18px;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)

st.title("🚨 Real-Time Seizure Risk Forecasting & Clinical Alarm Console")
st.caption("Dual-Threshold Hysteresis Early Warning Horizon Controller (15 - 30 Minutes Pre-Ictal Window)")

# Sidebar Controls
with st.sidebar:
    st.subheader("Patient & Alarm Controls")
    
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
        "Telemetry Source",
        ["Real Hospital EDF File (.edf)", "Dynamic Pre-Ictal Simulation"],
        index=0
    )
    
    state = "preictal"
    if "Simulation" in stream_source:
        state = st.selectbox("Simulated Phase", ["interictal", "preictal", "ictal"], index=1)
        
    alarm_threshold = st.slider("Alarm Trigger Threshold", 0.40, 0.90, 0.65, 0.05)
    clear_threshold = st.slider("Alarm Clearance Threshold", 0.20, 0.60, 0.40, 0.05)
    
    if st.button("Reset Telemetry History", use_container_width=True):
        st.session_state.forecast_history = []
        st.session_state.alarm_log = []
        st.rerun()

# Session State for History & Log
if "forecast_history" not in st.session_state:
    st.session_state.forecast_history = []
if "alarm_log" not in st.session_state:
    st.session_state.alarm_log = []

@st.cache_resource
def get_engine(a_thresh, c_thresh):
    ckpt = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
    eng = RealTimeForecastingEngine(
        model_checkpoint_path=str(ckpt),
        alarm_threshold=a_thresh,
        clear_threshold=c_thresh
    )
    sim = CHBMITDataSimulator()
    return eng, sim

engine, simulator = get_engine(alarm_threshold, clear_threshold)

# Ingest Data
if "Real" in stream_source:
    raw_eeg, _ = simulator.load_real_patient_recording(patient_id, duration_sec=12)
    source_tag = f"Real Clinical EDF Stream: {patient['edf_filename']}"
else:
    raw_eeg, _ = simulator.generate_patient_recording(patient_id, duration_sec=12, state=state)
    source_tag = f"Simulated {state.upper()} Dynamic State"

engine.ingest_sample_block(raw_eeg)
res = engine.process_latest_window()

# Append to history
st.session_state.forecast_history.append({
    "time_step": len(st.session_state.forecast_history) + 1,
    "smoothed_risk": res["smoothed_risk"],
    "raw_risk": res["raw_risk"],
    "alarm_active": res["alarm_active"]
})

if len(st.session_state.forecast_history) > 40:
    st.session_state.forecast_history = st.session_state.forecast_history[-40:]

# Log alarm if triggered
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
            "action": "Immediate Pre-Ictal Protocol: Administer Rescue Medication & Safety Hold"
        })

# Header Card
st.markdown(f"""
<div class="header-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h4 style="margin:0; color:#0284c7; font-weight:700;">{patient['display_name']}</h4>
            <span style="color:#64748b; font-size:0.9rem;">
                Primary Focus: <b style="color:#dc2626;">{patient['focal_zone']}</b> &bull; Prescribed AED: <b>{', '.join(patient['medications'])}</b>
            </span>
        </div>
        <span style="background-color:#e0f2fe; color:#0369a1; padding:4px 12px; border-radius:6px; font-weight:700; font-size:0.85rem; border:1px solid #bae6fd;">
            {source_tag}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Continuous Seizure Risk", f"{res['smoothed_risk']*100:.1f}%")
with col2:
    lead_display = f"{res['lead_time_min']:.1f} min" if res['lead_time_min'] else f"{patient['metrics']['mean_lead_time_min']:.1f} min"
    st.metric("Estimated Horizon", lead_display)
with col3:
    st.metric("Alarm Hysteresis State", "TRIGGERED" if res["alarm_active"] else "CLEAR")
with col4:
    st.metric("Historical Patient Sensitivity", f"{patient['metrics']['sensitivity']:.1f}%")

# Main Seizure Risk Forecasting Chart
df_hist = pd.DataFrame(st.session_state.forecast_history)

fig = go.Figure()

# Dual Threshold lines
fig.add_hline(
    y=alarm_threshold, line_dash="dash", line_color="#dc2626",
    annotation_text=f"Trigger Threshold ({alarm_threshold:.2f})",
    annotation_position="top right", annotation_font_color="#dc2626"
)
fig.add_hline(
    y=clear_threshold, line_dash="dash", line_color="#15803d",
    annotation_text=f"Reset Threshold ({clear_threshold:.2f})",
    annotation_position="bottom right", annotation_font_color="#15803d"
)

# Smoothed risk trajectory
fig.add_trace(go.Scatter(
    x=df_hist["time_step"],
    y=df_hist["smoothed_risk"],
    mode='lines+markers',
    name="Smoothed Seizure Risk",
    line=dict(color="#0284c7", width=3),
    marker=dict(size=6, color="#0369a1")
))

fig.update_layout(
    template="plotly_white",
    height=380,
    margin=dict(l=40, r=40, t=30, b=40),
    yaxis=dict(title="Seizure Probability", range=[0.0, 1.05], gridcolor="#e2e8f0"),
    xaxis=dict(title="Temporal Epoch Sequence (10s Windows)", gridcolor="#e2e8f0"),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Emergency Clinical Action Protocol
if res["alarm_active"]:
    st.markdown(f"""
    <div class="protocol-card">
        <h3 style="color:#b91c1c; margin-top:0; font-size:1.25rem;">
            🚨 PRE-ICTAL EMERGENCY CLINICAL PROTOCOL TRIGGERED
        </h3>
        <p style="color:#334155; margin-bottom:8px;">
            The model has detected sustained Phase Locking Value (PLV) synchrony across <b>{patient['focal_zone']}</b> exceeding the clinical threshold of <b>{alarm_threshold:.2f}</b>.
            Estimated clinical seizure onset within <b>{lead_display}</b>.
        </p>
        <ol style="color:#0f172a; line-height:1.7; margin-bottom:0; font-weight:600;">
            <li>Position patient into safe recumbent posture (prevent physical falls or head trauma).</li>
            <li>Prepare prescribed rescue medication: <i>{patient['medications'][0]}</i>.</li>
            <li>Alert bedside nursing staff and EEG Epilepsy Monitoring Unit (EMU) physician on duty.</li>
            <li>Ensure video-telemetry recording channels are unobstructed.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Historical Clinical Alarm Audit Log
st.divider()
st.subheader("📋 Clinical Alarm Event Audit Log")

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
    st.info("No clinical alarms triggered in the current telemetry window. Patient is in safe baseline state.")
