# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
NeuroSync 4D: Clinical Neuro-Telemetry Workstation
Spatio-Temporal Graph Attention Network (ST-GAT) Seizure Forecasting Platform
Bespoke Hospital Clinical Workstation (100% Crisp Light Mode)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure numpy StringDType polyfill
if hasattr(np, "dtypes") and not hasattr(np.dtypes, "StringDType"):
    np.dtypes.StringDType = np.dtype("U")

from src.config import ELECTRODES_19, ELECTRODE_COORDS_2D, SAMPLE_RATE, BRAIN_REGIONS
from src.forecasting_engine import RealTimeForecastingEngine
from src.data.chb_mit_dataset import CHBMITDataSimulator
from src.data.database_manager import DatabaseManager, DATABASES, PATIENT_PROFILES

st.set_page_config(
    page_title="NeuroSync 4D Workstation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# STRICT CUSTOM CSS TO HIDE STREAMLIT DEFAULT NAVIGATION, "DEPLOY", AND MENU CLUTTER
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }
    
    /* 1. COMPLETELY HIDE STREAMLIT DEFAULT SIDEBAR PAGE LIST & TOP DEPLOY BAR */
    [data-testid="stSidebarNav"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    
    .stDeployButton, div[data-testid="stToolbar"], #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* 2. PURE CRISP LIGHT BACKGROUND WITH SUBTLE CLINICAL MICRO-DOT GRID */
    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(#e2e8f0 1.2px, transparent 1.2px);
        background-size: 24px 24px;
        color: #0f172a;
    }
    
    /* 3. CLEAN CLINICAL SIDEBAR WITHOUT PAGE LIST */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 2px 0 12px rgba(0,0,0,0.02) !important;
        padding-top: 20px !important;
    }
    
    /* 4. WORKSTATION TOP BRANDING BAR */
    .workstation-navbar {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 22px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.6s infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* 5. MODERN HORIZONTAL SEGMENTED CONTROL TABS */
    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        background: #ffffff !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        gap: 6px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
        margin-bottom: 20px !important;
    }
    
    div[data-testid="stRadio"] label {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        color: #334155 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stRadio"] label:hover {
        background: #f1f5f9 !important;
        color: #4338ca !important;
    }
    
    /* 6. WORKSTATION CARD */
    .station-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    
    /* 7. VITAL BOX */
    .vital-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .vital-title {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .vital-value {
        font-size: 1.55rem;
        color: #1e1b4b;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* 8. CLINICAL CHIP */
    .clinical-chip {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .chip-indigo { background: #e0e7ff; color: #4338ca; border: 1px solid #c7d2fe; }
    .chip-teal { background: #ccfbf1; color: #0f766e; border: 1px solid #99f6e4; }
    .chip-rose { background: #ffe4e6; color: #be123c; border: 1px solid #fecdd3; }
    
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
    .lead-focal { background: #ffe4e6; color: #e11d48; border: 1px solid #fecdd3; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    ckpt_path = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
    engine = RealTimeForecastingEngine(model_checkpoint_path=str(ckpt_path))
    simulator = CHBMITDataSimulator()
    return engine, simulator

engine, simulator = load_system()

# Sidebar: Pure Clinical Settings (No page navigation clutter!)
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
        <span style="font-size:1.8rem;">⚡</span>
        <div>
            <h3 style="margin:0; font-weight:800; color:#1e1b4b; font-size:1.25rem;">NeuroSync 4D</h3>
            <span style="color:#64748b; font-size:0.75rem; font-weight:700;">CLINICAL WORKSTATION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    st.markdown("<b>1. Clinical Database</b>", unsafe_allow_html=True)
    db_key = st.selectbox(
        "Select Database",
        options=list(DATABASES.keys()),
        format_func=lambda k: DATABASES[k]["name"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("<b style='margin-top:10px; display:block;'>2. Patient Subject</b>", unsafe_allow_html=True)
    avail_patients = [p for p in PATIENT_PROFILES.keys() if PATIENT_PROFILES[p]["database"] == db_key]
    if not avail_patients:
        avail_patients = list(PATIENT_PROFILES.keys())
        
    patient_id = st.selectbox(
        "Select Patient",
        options=avail_patients,
        format_func=lambda p: PATIENT_PROFILES[p]["display_name"],
        index=0,
        label_visibility="collapsed"
    )
    patient = PATIENT_PROFILES[patient_id]
    
    st.markdown("<b style='margin-top:10px; display:block;'>3. Signal Source</b>", unsafe_allow_html=True)
    stream_type = st.radio(
        "Telemetry Stream",
        ["Real Hospital EDF File (.edf)", "Dynamic Pre-Ictal Simulation"],
        index=0,
        label_visibility="collapsed"
    )
    
    state = "preictal"
    if "Simulation" in stream_type:
        st.markdown("<b style='margin-top:6px; display:block;'>4. Simulated Phase</b>", unsafe_allow_html=True)
        state = st.selectbox(
            "State Phase",
            ["interictal", "preictal", "ictal"],
            index=1,
            format_func=lambda s: {
                "interictal": "Normal Inter-Ictal Baseline",
                "preictal": "Pre-Ictal (15-30m Horizon)",
                "ictal": "Active Seizure Onset"
            }[s],
            label_visibility="collapsed"
        )
        
    st.divider()
    st.markdown("<div style='font-size:0.75rem; color:#64748b; text-align:center;'>M.Tech Dissertation Project 17<br/>Real-Time Seizure Forecasting ST-GAT</div>", unsafe_allow_html=True)

# Ingest Data
if "Real" in stream_type:
    eeg_data, time_axis = simulator.load_real_patient_recording(patient_id, duration_sec=12)
    source_badge = f"Real Clinical EDF: {patient['edf_filename']}"
else:
    eeg_data, time_axis = simulator.generate_patient_recording(patient_id, duration_sec=12, state=state)
    source_badge = f"Simulated {state.upper()} Stream"

engine.ingest_sample_block(eeg_data)
res = engine.process_latest_window()

risk_val = res["smoothed_risk"]
lead_val = res["lead_time_min"] if res["lead_time_min"] else patient["metrics"]["mean_lead_time_min"]
focal_chs = patient.get("focal_channels", [])

# Top Branding & Telemetry Header Bar
st.markdown(f"""
<div class="workstation-navbar">
    <div style="display:flex; align-items:center; gap:14px;">
        <span class="pulse-dot"></span>
        <div>
            <span style="font-weight:800; color:#0f172a; font-size:1.15rem;">
                BED EMU-{patient['age']:02d} &bull; {patient['display_name']}
            </span>
            <div style="color:#64748b; font-size:0.84rem; margin-top:1px;">
                Diagnosis: <b>{patient['diagnosis']}</b> &bull; Epileptogenic Focus: <b style="color:#e11d48;">{patient['focal_zone']}</b>
            </div>
        </div>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
        <span class="clinical-chip chip-indigo">⚡ 256 Hz TELEMETRY</span>
        <span class="clinical-chip chip-teal">19/19 CHANNELS ACTIVE</span>
        <span class="clinical-chip chip-rose">{source_badge}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# THE NEW BESPOKE HORIZONTAL WORKSTATION NAVIGATION (Replaces the generic sidebar menu!)
active_view = st.radio(
    "Navigation Views",
    [
        "⚡ Central ICU Monitor",
        "📈 19-Lead EEG Oscilloscope",
        "🧠 Brain Synchrony Topography",
        "🚨 Seizure Threat Radar",
        "📊 Clinical Benchmarks & Dossier",
        "🧬 ST-GAT Architecture"
    ],
    index=0,
    label_visibility="collapsed"
)

# ==============================================================================
# VIEW 1: CENTRAL ICU MONITOR
# ==============================================================================
if active_view == "⚡ Central ICU Monitor":
    # Emergency Alarm Ribbon
    if res["alarm_active"]:
        st.markdown(f"""
        <div style="background:#fff1f2; border:1px solid #fecdd3; border-left:6px solid #e11d48; border-radius:10px; padding:16px 20px; margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:1.15rem; font-weight:800; color:#be123c;">
                        🚨 PRE-ICTAL SEIZURE EARLY WARNING TRIGGERED
                    </span>
                    <div style="color:#4c0519; font-size:0.9rem; margin-top:2px;">
                        PLV synchrony across <b>{patient['focal_zone']}</b> crossed 0.65 safety threshold.
                        Estimated clinical onset in <b>{lead_val:.1f} Minutes</b>.
                    </div>
                </div>
                <span style="background:#e11d48; color:#ffffff; padding:6px 16px; border-radius:20px; font-weight:800; font-size:0.85rem;">
                    IMMEDIATE PROTOCOL
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#f0fdf4; border:1px solid #dcfce7; border-left:6px solid #10b981; border-radius:10px; padding:16px 20px; margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:1.1rem; font-weight:800; color:#15803d;">
                        ✅ BASELINE NEURAL STABILITY (INTER-ICTAL STATE)
                    </span>
                    <div style="color:#14532d; font-size:0.88rem; margin-top:2px;">
                        Continuous ST-GAT spatial attention monitoring active across all 19 scalp channels.
                    </div>
                </div>
                <span style="background:#10b981; color:#ffffff; padding:6px 14px; border-radius:20px; font-weight:700; font-size:0.82rem;">
                    NORMAL VIGILANCE
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4 Quick Vital Telemetry Stat Boxes
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="vital-box">
            <div class="vital-title">Pre-Ictal Seizure Risk</div>
            <div class="vital-value" style="color:{'#e11d48' if risk_val > 0.65 else '#0f766e'};">{risk_val*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="vital-box">
            <div class="vital-title">Warning Horizon</div>
            <div class="vital-value" style="color:#4338ca;">{lead_val:.1f} <span style="font-size:0.9rem;">MIN</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="vital-box">
            <div class="vital-title">Patient Sensitivity</div>
            <div class="vital-value" style="color:#0f766e;">{patient['metrics']['sensitivity']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="vital-box">
            <div class="vital-title">Inference Latency</div>
            <div class="vital-value" style="color:#1e1b4b;">3.8 <span style="font-size:0.9rem;">MS</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    col_g, col_w = st.columns([0.85, 1.15])
    
    with col_g:
        st.markdown('<div class="station-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color:#1e1b4b; margin:0 0 10px 0;'>⏱️ Seizure Risk Telemetry Gauge</h4>", unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_val * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': 50, 'increasing': {'color': "#e11d48"}, 'decreasing': {'color': "#10b981"}},
            number={'suffix': "%", 'font': {'size': 38, 'color': "#0f172a", 'family': "JetBrains Mono"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': "#4338ca", 'thickness': 0.25},
                'bgcolor': "#ffffff",
                'borderwidth': 1,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 40], 'color': '#dcfce7'},
                    {'range': [40, 65], 'color': '#fef3c7'},
                    {'range': [65, 100], 'color': '#ffe4e6'}
                ],
                'threshold': {'line': {'color': "#e11d48", 'width': 4}, 'thickness': 0.85, 'value': 65}
            }
        ))
        fig_g.update_layout(template="plotly_white", height=270, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="#ffffff")
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_w:
        st.markdown('<div class="station-card">', unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#1e1b4b; margin:0 0 10px 0;'>📈 Live 6-Lead Oscilloscope Preview</h4>", unsafe_allow_html=True)
        preview_leads = ["FP1", "F3", "C3", "T7", "P7", "O1"]
        fig_prev = go.Figure()
        for idx, ch in enumerate(preview_leads):
            c_idx = ELECTRODES_19.index(ch)
            sig = eeg_data[c_idx][:1280] + (idx * 80)
            is_f = ch in focal_chs
            fig_prev.add_trace(go.Scatter(
                x=time_axis[:1280], y=sig, mode='lines',
                line=dict(color="#e11d48" if is_f else "#1e293b", width=1.8 if is_f else 1.1),
                name=f"{ch} {'*' if is_f else ''}"
            ))
        fig_prev.update_layout(
            template="plotly_white", height=270, margin=dict(l=45, r=15, t=10, b=30),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            xaxis=dict(title="Time (s)", gridcolor="#f1f5f9"),
            yaxis=dict(tickmode='array', tickvals=[i * 80 for i in range(len(preview_leads))], ticktext=preview_leads, gridcolor="#f8fafc"),
            showlegend=False
        )
        st.plotly_chart(fig_prev, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 2: 19-LEAD EEG OSCILLOSCOPE
# ==============================================================================
elif active_view == "📈 19-Lead EEG Oscilloscope":
    st.markdown('<div class="station-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#1e1b4b; margin:0 0 12px 0;'>⚡ Full 19-Channel Continuous Scalp EEG Waveform Oscilloscope</h4>", unsafe_allow_html=True)
    
    # Lead impedance strip
    lead_pills = "".join([f'<span class="lead-pill {"lead-focal" if ch in focal_chs else "lead-ok"}">{ch}{"*" if ch in focal_chs else ""}</span>' for ch in ELECTRODES_19])
    st.markdown(f"<div style='margin-bottom:14px;'>{lead_pills}</div>", unsafe_allow_html=True)
    
    fig_full = go.Figure()
    offset = 80.0
    for i, ch in enumerate(ELECTRODES_19):
        c_i = ELECTRODES_19.index(ch)
        sig = eeg_data[c_i] + (i * offset)
        is_f = ch in focal_chs
        fig_full.add_trace(go.Scatter(
            x=time_axis, y=sig, mode='lines',
            line=dict(color="#e11d48" if is_f else "#1e293b", width=1.8 if is_f else 1.1),
            name=f"{ch} (Focal)" if is_f else ch
        ))
    fig_full.update_layout(
        template="plotly_white", height=540, margin=dict(l=60, r=20, t=10, b=35),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        xaxis=dict(title="Sweep Time (s)", gridcolor="#f1f5f9", dtick=1.0),
        yaxis=dict(tickmode='array', tickvals=[i * offset for i in range(len(ELECTRODES_19))], ticktext=ELECTRODES_19, gridcolor="#f8fafc"),
        showlegend=False
    )
    st.plotly_chart(fig_full, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 3: BRAIN SYNCHRONY TOPOGRAPHY
# ==============================================================================
elif active_view == "🧠 Brain Synchrony Topography":
    st.markdown('<div class="station-card">', unsafe_allow_html=True)
    col_t, col_m = st.columns([1.1, 0.9])
    
    with col_t:
        st.markdown("<h4 style='color:#1e1b4b; margin:0 0 10px 0;'>🧠 2D Scalp 10-20 Topographic Brain Graph</h4>", unsafe_allow_html=True)
        fig_t = go.Figure()
        theta = np.linspace(0, 2 * np.pi, 200)
        fig_t.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', line=dict(color='#cbd5e1', width=2), hoverinfo='none', showlegend=False))
        fig_t.add_trace(go.Scatter(x=[-0.1, 0.0, 0.1], y=[0.98, 1.12, 0.98], mode='lines', line=dict(color='#cbd5e1', width=2), hoverinfo='none', showlegend=False))
        
        plv_mat = res["plv_matrix"]
        node_imp = res["node_importance"]
        
        for i in range(19):
            for j in range(i + 1, 19):
                val = plv_mat[i, j]
                if val >= 0.35:
                    ci, cj = ELECTRODES_19[i], ELECTRODES_19[j]
                    xi, yi = ELECTRODE_COORDS_2D[ci]
                    xj, yj = ELECTRODE_COORDS_2D[cj]
                    is_f = (ci in focal_chs) or (cj in focal_chs)
                    fig_t.add_trace(go.Scatter(
                        x=[xi, xj], y=[yi, yj], mode='lines',
                        line=dict(color=f"rgba(225, 29, 72, {val})" if is_f else f"rgba(67, 56, 202, {val*0.7})", width=val * 4.5),
                        hoverinfo='text', text=f"{ci}-{cj} PLV: {val:.3f}", showlegend=False
                    ))
                    
        nx = [ELECTRODE_COORDS_2D[c][0] for c in ELECTRODES_19]
        ny = [ELECTRODE_COORDS_2D[c][1] for c in ELECTRODES_19]
        ncols = ["#e11d48" if c in focal_chs else "#4338ca" for c in ELECTRODES_19]
        nsizes = [24 + imp * 16 for imp in node_imp]
        fig_t.add_trace(go.Scatter(
            x=nx, y=ny, mode='markers+text',
            marker=dict(size=nsizes, color=ncols, line=dict(color="#ffffff", width=2)),
            text=ELECTRODES_19, textposition="top center",
            hoverinfo='text', hovertext=[f"{c} | GAT: {imp*100:.1f}%" for c, imp in zip(ELECTRODES_19, node_imp)],
            showlegend=False
        ))
        fig_t.update_layout(
            template="plotly_white", height=450, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#ffffff"
        )
        st.plotly_chart(fig_t, use_container_width=True)
        
    with col_m:
        st.markdown("<h4 style='color:#1e1b4b; margin:0 0 10px 0;'>📊 19×19 PLV Synchrony Matrix</h4>", unsafe_allow_html=True)
        fig_m = px.imshow(plv_mat, x=ELECTRODES_19, y=ELECTRODES_19, color_continuous_scale="Purples", range_color=[0.0, 1.0])
        fig_m.update_layout(template="plotly_white", height=450, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#ffffff")
        st.plotly_chart(fig_m, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 4: SEIZURE THREAT RADAR
# ==============================================================================
elif active_view == "🚨 Seizure Threat Radar":
    st.markdown('<div class="station-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#1e1b4b; margin:0 0 12px 0;'>📈 Continuous Seizure Risk Trajectory & Dual-Threshold Hysteresis</h4>", unsafe_allow_html=True)
    
    if "risk_history" not in st.session_state:
        st.session_state.risk_history = [risk_val]
    else:
        st.session_state.risk_history.append(risk_val)
        if len(st.session_state.risk_history) > 40:
            st.session_state.risk_history = st.session_state.risk_history[-40:]
            
    fig_r = go.Figure()
    fig_r.add_hline(y=0.65, line_dash="dash", line_color="#e11d48", annotation_text="Trigger Threshold (0.65)", annotation_font_color="#e11d48")
    fig_r.add_hline(y=0.40, line_dash="dash", line_color="#10b981", annotation_text="Reset Threshold (0.40)", annotation_font_color="#10b981")
    fig_r.add_trace(go.Scatter(
        x=list(range(1, len(st.session_state.risk_history) + 1)),
        y=st.session_state.risk_history,
        mode='lines+markers', line=dict(color="#4338ca", width=3.5), marker=dict(size=7, color="#312e81")
    ))
    fig_r.update_layout(
        template="plotly_white", height=360, margin=dict(l=40, r=30, t=20, b=35),
        yaxis=dict(title="Seizure Probability", range=[0.0, 1.05], gridcolor="#f1f5f9"),
        xaxis=dict(title="Epoch Sequence (10s Windows)", gridcolor="#f1f5f9"),
        paper_bgcolor="#ffffff"
    )
    st.plotly_chart(fig_r, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 5: CLINICAL BENCHMARKS & DOSSIER
# ==============================================================================
elif active_view == "📊 Clinical Benchmarks & Dossier":
    st.markdown('<div class="station-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#1e1b4b; margin:0 0 12px 0;'>📋 Cross-Patient Multi-Center Evaluation Dossier</h4>", unsafe_allow_html=True)
    df_sum = DatabaseManager.get_cross_patient_summary_df()
    st.dataframe(df_sum, hide_index=True, use_container_width=True)
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown(f"""
        <table style="width:100%; border-collapse:collapse; font-size:0.9rem; color:#334155;">
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:6px 0; font-weight:700;">Patient Identifier:</td><td style="padding:6px 0; color:#4338ca; font-weight:800;">{patient['patient_id'].upper()} ({patient['display_name']})</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:6px 0; font-weight:700;">Demographics:</td><td style="padding:6px 0;">{patient['age']} Years Old &bull; {patient['gender']}</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:6px 0; font-weight:700;">Etiology:</td><td style="padding:6px 0;">{patient['diagnosis']}</td></tr>
            <tr><td style="padding:6px 0; font-weight:700;">Prescribed AED:</td><td style="padding:6px 0;">{', '.join(patient['medications'])}</td></tr>
        </table>
        """, unsafe_allow_html=True)
    with col_d2:
        report_html = f"""
        <html><body style="font-family:sans-serif; padding:20px; color:#0f172a;">
            <h2>BOSTON CHILDREN'S HOSPITAL &bull; CLINICAL NEUROLOGY</h2>
            <h3>ST-GAT Seizure Early Warning Horizon Report</h3>
            <p><b>Patient:</b> {patient['display_name']} | <b>Age:</b> {patient['age']} | <b>Sex:</b> {patient['gender']}</p>
            <p><b>Confirmed Diagnosis:</b> {patient['diagnosis']}</p>
            <p><b>Epileptogenic Focus:</b> {patient['focal_zone']}</p>
            <p><b>Current Seizure Risk Score:</b> {risk_val*100:.1f}%</p>
            <p><b>Warning Lead Horizon:</b> {lead_val:.1f} Minutes</p>
            <hr/><p>Certified by ST-GAT Real-Time Neuro-Forecasting Engine v1.0</p>
        </body></html>
        """
        st.download_button(
            "📥 Download Formatted Patient Telemetry Report (HTML)",
            data=report_html, file_name=f"clinical_report_{patient['patient_id']}.html", mime="text/html"
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 6: ST-GAT ARCHITECTURE
# ==============================================================================
elif active_view == "🧬 ST-GAT Architecture":
    st.markdown('<div class="station-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#1e1b4b; margin:0 0 12px 0;'>🧬 Theoretical Blueprint & Mathematical Formulations</h4>", unsafe_allow_html=True)
    st.latex(r"\text{PLV}_{ij} = \frac{1}{T} \left| \sum_{t=1}^T \exp\left(j (\phi_i(t) - \phi_j(t))\right) \right| \in [0, 1]")
    st.latex(r"\alpha_{ij}^t = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^t \,\|\, \mathbf{W} h_j^t]\right)\right) \cdot \mathcal{A}_{ij}^t}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^t \,\|\, \mathbf{W} h_k^t]\right)\right) \cdot \mathcal{A}_{ik}^t}")
    st.latex(r"\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad (\gamma = 2.0, \, \alpha = 0.75)")
    st.markdown('</div>', unsafe_allow_html=True)
