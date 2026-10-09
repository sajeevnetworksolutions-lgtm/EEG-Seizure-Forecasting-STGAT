"""
Page 3 - Real-Time Seizure Risk Forecasting & Clinical Alarm Engine
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHB_MIT_PATIENTS
from src.forecasting_engine import RealTimeForecastingEngine
from src.data.chb_mit_dataset import CHBMITDataSimulator

st.set_page_config(page_title="Seizure Forecasting", page_icon="🚨", layout="wide")

st.title("🚨 Real-Time Seizure Risk Forecasting & Alarm Console")
st.caption("Dual-Threshold Hysteresis Early Warning Horizon Controller (15 - 30 Minutes Pre-Ictal Window)")

with st.sidebar:
    st.subheader("Simulation Hub")
    patient = st.selectbox("Patient", CHB_MIT_PATIENTS, index=0)
    state = st.selectbox("State", ["interictal", "preictal", "ictal"], index=1)
    alarm_threshold = st.slider("Alarm Trigger Threshold", 0.40, 0.90, 0.65, 0.05)
    clear_threshold = st.slider("Alarm Clearance Threshold", 0.20, 0.60, 0.40, 0.05)

if "forecast_history" not in st.session_state:
    st.session_state.forecast_history = []

@st.cache_resource
def get_engine():
    ckpt = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt.exists():
        ckpt = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
    eng = RealTimeForecastingEngine(model_checkpoint_path=str(ckpt), alarm_threshold=alarm_threshold, clear_threshold=clear_threshold)
    sim = CHBMITDataSimulator()
    return eng, sim

engine, simulator = get_engine()
engine.alarm_threshold = alarm_threshold
engine.clear_threshold = clear_threshold

# Step simulation button
col_btn, _ = st.columns([1, 4])
with col_btn:
    step_clicked = st.button("Simulate Next Second (+1s)", use_container_width=True)

# Generate data
raw_eeg, _ = simulator.generate_patient_recording(patient, duration_sec=10, state=state)
engine.ingest_sample_block(raw_eeg)
res = engine.process_latest_window()

# Append to history
curr_t = len(st.session_state.forecast_history) + 1
entry = {
    "Step": curr_t,
    "Time": f"{curr_t}s",
    "Risk": res["smoothed_risk"] * 100.0,
    "Alarm": "ACTIVE" if res["alarm_active"] else "NORMAL",
    "Status": res["status_label"],
    "Lead Time (min)": f"{res['lead_time_min']:.1f}" if res['lead_time_min'] else "N/A"
}
st.session_state.forecast_history.append(entry)

# Top Status Indicator
status_bg = res["status_color"]
lead_str = f"| Lead Time: {res['lead_time_min']:.1f} Min" if res['lead_time_min'] else ""
st.markdown(f"""
<div style="background-color: {status_bg}22; border: 2px solid {status_bg}; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px;">
    <span style="font-size: 1.3rem; font-weight: 700; color: {status_bg};">
        STATUS: {res['status_label']} {lead_str}
    </span>
</div>
""", unsafe_allow_html=True)

# Risk Probability Trajectory Plot
df_hist = pd.DataFrame(st.session_state.forecast_history)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_hist["Step"], y=df_hist["Risk"],
    mode="lines+markers",
    line=dict(color="#38bdf8", width=2.5),
    name="Pre-Ictal Risk Score (%)"
))

# Threshold lines
fig.add_hline(y=alarm_threshold*100, line_dash="dash", line_color="#ef4444", annotation_text=f"Alarm Trigger ({alarm_threshold*100:.0f}%)")
fig.add_hline(y=clear_threshold*100, line_dash="dot", line_color="#10b981", annotation_text=f"Clearance Reset ({clear_threshold*100:.0f}%)")

fig.update_layout(
    height=360,
    margin=dict(l=30, r=20, t=20, b=30),
    plot_bgcolor="#0b132b",
    paper_bgcolor="#1e293b",
    font=dict(color="#94a3b8"),
    xaxis=dict(title="Monitoring Time Step (Seconds)", showgrid=True, gridcolor="#1e293b"),
    yaxis=dict(title="Pre-Ictal Risk Probability (%)", range=[0, 105], showgrid=True, gridcolor="#1e293b")
)
st.plotly_chart(fig, use_container_width=True)

# Alarm Log Table & CSV Export
st.subheader("Clinical Alarm History Log")
st.dataframe(df_hist.tail(10), use_container_width=True, hide_index=True)

csv = df_hist.to_csv(index=False).encode('utf-8')
st.download_button("Download Alarm History (CSV)", data=csv, file_name="forecasting_alarm_log.csv", mime="text/csv")
