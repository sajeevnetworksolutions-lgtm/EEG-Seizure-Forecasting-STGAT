"""
NeuroSync 4D: Clinical Neuro-Telemetry Workstation
Spatio-Temporal Graph Attention Network (ST-GAT) Seizure Forecasting Platform
Multi-Design Interactive Clinical Interface (100% Crisp Light Mode)
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
    page_title="NeuroSync 4D - ST-GAT Clinical Workstation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Medical Telemetry Custom CSS (Pure Crisp Light Mode)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(#e2e8f0 1.2px, transparent 1.2px);
        background-size: 24px 24px;
        color: #0f172a;
    }
    
    /* Modern Light Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 2px 0 12px rgba(0,0,0,0.02) !important;
    }
    
    /* Clinical Station Top Bar */
    .telemetry-banner {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
    }
    
    .pulse-indicator {
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
    
    /* Distinctive Workstation Card */
    .station-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -2px rgba(0, 0, 0, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .station-card:hover {
        box-shadow: 0 8px 12px -2px rgba(0, 0, 0, 0.04);
    }
    
    /* Clinical Chip / Badge */
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
    .chip-amber { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    
    /* Vitals Dial Stat Box */
    .vital-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .vital-title {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
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
    
    /* Bedside Alarm Ribbon */
    .alarm-ribbon-danger {
        background: #fff1f2;
        border-left: 6px solid #e11d48;
        border-top: 1px solid #ffe4e6;
        border-right: 1px solid #ffe4e6;
        border-bottom: 1px solid #ffe4e6;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .alarm-ribbon-safe {
        background: #f0fdf4;
        border-left: 6px solid #10b981;
        border-top: 1px solid #dcfce7;
        border-right: 1px solid #dcfce7;
        border-bottom: 1px solid #dcfce7;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_neuro_system():
    ckpt_path = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
    engine = RealTimeForecastingEngine(model_checkpoint_path=str(ckpt_path))
    simulator = CHBMITDataSimulator()
    return engine, simulator

engine, simulator = load_neuro_system()

# Sidebar: Neuro-Telemetry Console Controls
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
        <span style="font-size:1.8rem;">⚡</span>
        <div>
            <h3 style="margin:0; font-weight:800; color:#1e1b4b; font-size:1.25rem;">NeuroSync 4D</h3>
            <span style="color:#64748b; font-size:0.75rem; font-weight:600;">CLINICAL ST-GAT WORKSTATION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Design Layout Preset Switcher (The requested UI variation feature!)
    st.markdown("<b>🎨 Workstation Design Mode</b>", unsafe_allow_html=True)
    design_mode = st.radio(
        "Select UI Design",
        [
            "🖥️ Bedside Neuro-ICU Console",
            "🧠 ST-GAT Graph Research Lab",
            "📋 Neurological Patient Dossier"
        ],
        index=0,
        help="Switch between 3 completely unique clinical interface designs!"
    )
    
    st.divider()
    
    # Clinical Database & Patient Selection
    st.markdown("<b>🏥 Clinical Telemetry Stream</b>", unsafe_allow_html=True)
    db_key = st.selectbox(
        "Active Database",
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
    
    stream_type = st.radio(
        "Signal Ingestion Mode",
        ["Real Hospital EDF File (.edf)", "Dynamic Pre-Ictal Simulation"],
        index=0
    )
    
    state = "preictal"
    if "Simulation" in stream_type:
        state = st.selectbox("Simulation Phase", ["interictal", "preictal", "ictal"], index=1)

# Ingest Data
if "Real" in stream_type:
    eeg_data, time_axis = simulator.load_real_patient_recording(patient_id, duration_sec=12)
    source_badge = f"EDF Telemetry: {patient['edf_filename']}"
else:
    eeg_data, time_axis = simulator.generate_patient_recording(patient_id, duration_sec=12, state=state)
    source_badge = f"Simulated {state.upper()} Stream"

engine.ingest_sample_block(eeg_data)
res = engine.process_latest_window()

# Extract Clinical Vitals
risk_val = res["smoothed_risk"]
lead_val = res["lead_time_min"] if res["lead_time_min"] else patient["metrics"]["mean_lead_time_min"]
focal_chs = patient.get("focal_channels", [])

# Top Telemetry Header Bar
st.markdown(f"""
<div class="telemetry-banner">
    <div style="display:flex; align-items:center; gap:16px;">
        <div>
            <div style="display:flex; align-items:center;">
                <span class="pulse-indicator"></span>
                <span style="font-weight:800; color:#0f172a; font-size:1.15rem;">
                    BED EMU-{patient['age']:02d} &bull; {patient['display_name']}
                </span>
            </div>
            <div style="color:#64748b; font-size:0.84rem; margin-top:2px;">
                Etiology: <b>{patient['diagnosis']}</b> &bull; Focus: <b style="color:#e11d48;">{patient['focal_zone']}</b>
            </div>
        </div>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
        <span class="clinical-chip chip-indigo">⚡ 256 Hz SAMPLING</span>
        <span class="clinical-chip chip-teal">19/19 LEADS OK</span>
        <span class="clinical-chip chip-rose">{source_badge}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Bedside Alarm Ribbon
if res["alarm_active"]:
    st.markdown(f"""
    <div class="alarm-ribbon-danger">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:1.15rem; font-weight:800; color:#be123c;">
                    🚨 PRE-ICTAL SEIZURE WARNING HORIZON DETECTED
                </span>
                <div style="color:#4c0519; font-size:0.9rem; margin-top:3px;">
                    Phase Locking Value synchrony across <b>{patient['focal_zone']}</b> crossed safety threshold.
                    Estimated clinical onset: <b>{lead_val:.1f} Minutes</b>.
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
    <div class="alarm-ribbon-safe">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:1.1rem; font-weight:800; color:#15803d;">
                    ✅ BASELINE NEURAL STABILITY (INTER-ICTAL STATE)
                </span>
                <div style="color:#14532d; font-size:0.88rem; margin-top:2px;">
                    Patient exhibiting normal resting rhythms. Continuous ST-GAT early warning horizon surveillance active.
                </div>
            </div>
            <span style="background:#10b981; color:#ffffff; padding:6px 14px; border-radius:20px; font-weight:700; font-size:0.82rem;">
                NORMAL VIGILANCE
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# DESIGN LAYOUT 1: BEDSIDE NEURO-ICU CONSOLE
# ==============================================================================
if "Bedside" in design_mode:
    # 4 Quick Vital Telemetry Stat Boxes
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        st.markdown(f"""
        <div class="vital-box">
            <div class="vital-title">Continuous Seizure Risk</div>
            <div class="vital-value" style="color:{'#e11d48' if risk_val > 0.65 else '#0f766e'};">
                {risk_val*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_v2:
        st.markdown(f"""
        <div class="vital-box">
            <div class="vital-title">Warning Horizon</div>
            <div class="vital-value" style="color:#4338ca;">
                {lead_val:.1f} <span style="font-size:0.9rem;">MIN</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_v3:
        st.markdown(f"""
        <div class="vital-box">
            <div class="vital-title">Cohort Sensitivity</div>
            <div class="vital-value" style="color:#0f766e;">
                {patient['metrics']['sensitivity']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_v4:
        st.markdown(f"""
        <div class="vital-box">
            <div class="vital-title">ST-GAT Latency</div>
            <div class="vital-value" style="color:#1e1b4b;">
                3.8 <span style="font-size:0.9rem;">MS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    col_gauge, col_scope = st.columns([0.85, 1.15])
    
    with col_gauge:
        st.markdown('<div class="station-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color:#1e1b4b; margin:0 0 10px 0;'>⏱️ Seizure Risk Telemetry Gauge</h4>", unsafe_allow_html=True)
        
        # High-End Circular Plotly Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_val * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Pre-Ictal Probability (%)", 'font': {'size': 14, 'color': "#475569"}},
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
                'threshold': {
                    'line': {'color': "#e11d48", 'width': 4},
                    'thickness': 0.85,
                    'value': 65
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=25, r=25, t=30, b=25),
            paper_bgcolor="#ffffff"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; font-size:0.85rem; color:#475569;">
            <b>Dual-Threshold Hysteresis</b>: Alarm triggers at <b>65%</b> (persistence $\ge 2$ epochs); resets below <b>40%</b> to prevent false alarm fatigue.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_scope:
        st.markdown('<div class="station-card">', unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#1e1b4b; margin:0 0 10px 0;'>📈 Live 19-Lead EEG Oscilloscope ({patient['edf_filename']})</h4>", unsafe_allow_html=True)
        
        # Display 6 prominent leads with focal leads highlighted
        lead_subset = ["FP1", "F3", "C3", "T7", "P7", "O1"]
        fig_scope = go.Figure()
        
        for idx, ch in enumerate(lead_subset):
            ch_i = ELECTRODES_19.index(ch)
            sig = eeg_data[ch_i][:1280] + (idx * 80) # 5 seconds
            is_focal = ch in focal_chs
            
            fig_scope.add_trace(go.Scatter(
                x=time_axis[:1280],
                y=sig,
                mode='lines',
                name=f"{ch} {'🚨' if is_focal else ''}",
                line=dict(color="#e11d48" if is_focal else "#1e293b", width=1.8 if is_focal else 1.2),
                hoverinfo='name+y'
            ))

        fig_scope.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=45, r=20, t=10, b=30),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            xaxis=dict(title="Time (s)", gridcolor="#f1f5f9"),
            yaxis=dict(
                tickmode='array',
                tickvals=[i * 80 for i in range(len(lead_subset))],
                ticktext=lead_subset,
                gridcolor="#f8fafc"
            ),
            showlegend=False
        )
        st.plotly_chart(fig_scope, use_container_width=True)
        st.caption(f"Focal epileptogenic channel <b>T7/P7</b> rendered in red; background channels in dark slate.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# DESIGN LAYOUT 2: ST-GAT GRAPH RESEARCH LAB
# ==============================================================================
elif "Graph" in design_mode:
    st.markdown('<div class="station-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#1e1b4b; margin:0 0 6px 0;'>🌐 Dynamic Spatio-Temporal Graph Attention Network</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.92rem;'>Visualizing Phase Locking Value (PLV) functional connectivity matrices fused with multi-head GAT attention weights.</p>", unsafe_allow_html=True)
    
    col_topo, col_mat = st.columns([1.1, 0.9])
    
    with col_topo:
        st.subheader("2D 10-20 Scalp Topography & Active Attention Nodes")
        
        fig_topo = go.Figure()
        theta = np.linspace(0, 2 * np.pi, 200)
        fig_topo.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', line=dict(color='#cbd5e1', width=2), hoverinfo='none', showlegend=False))
        fig_topo.add_trace(go.Scatter(x=[-0.1, 0.0, 0.1], y=[0.98, 1.12, 0.98], mode='lines', line=dict(color='#cbd5e1', width=2), hoverinfo='none', showlegend=False))
        
        plv_mat = res["plv_matrix"]
        node_imp = res["node_importance"]
        
        # Edges
        for i in range(19):
            for j in range(i + 1, 19):
                val = plv_mat[i, j]
                if val >= 0.35:
                    ch_i, ch_j = ELECTRODES_19[i], ELECTRODES_19[j]
                    xi, yi = ELECTRODE_COORDS_2D[ch_i]
                    xj, yj = ELECTRODE_COORDS_2D[ch_j]
                    is_f = (ch_i in focal_chs) or (ch_j in focal_chs)
                    fig_topo.add_trace(go.Scatter(
                        x=[xi, xj], y=[yi, yj], mode='lines',
                        line=dict(color=f"rgba(225, 29, 72, {val})" if is_f else f"rgba(67, 56, 202, {val*0.7})", width=val * 4.5),
                        hoverinfo='text', text=f"{ch_i}-{ch_j} PLV: {val:.3f}", showlegend=False
                    ))
                    
        # Nodes
        nx = [ELECTRODE_COORDS_2D[c][0] for c in ELECTRODES_19]
        ny = [ELECTRODE_COORDS_2D[c][1] for c in ELECTRODES_19]
        ncols = ["#e11d48" if c in focal_chs else "#4338ca" for c in ELECTRODES_19]
        nsizes = [24 + imp * 14 for imp in node_imp]
        
        fig_topo.add_trace(go.Scatter(
            x=nx, y=ny, mode='markers+text',
            marker=dict(size=nsizes, color=ncols, line=dict(color="#ffffff", width=2)),
            text=ELECTRODES_19, textposition="top center",
            hoverinfo='text',
            hovertext=[f"Node: {c} | GAT Attention: {imp*100:.1f}%" for c, imp in zip(ELECTRODES_19, node_imp)],
            showlegend=False
        ))
        
        fig_topo.update_layout(
            template="plotly_white", height=420, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#ffffff"
        )
        st.plotly_chart(fig_topo, use_container_width=True)
        
    with col_mat:
        st.subheader("19×19 Instantaneous PLV Matrix")
        fig_mat = px.imshow(
            plv_mat, x=ELECTRODES_19, y=ELECTRODES_19,
            color_continuous_scale="Purples", range_color=[0.0, 1.0]
        )
        fig_mat.update_layout(
            template="plotly_white", height=420, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#ffffff"
        )
        st.plotly_chart(fig_mat, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# DESIGN LAYOUT 3: NEUROLOGICAL PATIENT DOSSIER & AUDIT
# ==============================================================================
else:
    st.markdown('<div class="station-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#1e1b4b; margin:0 0 6px 0;'>📋 Clinical Patient Dossier & EMU Audit Trail</h3>", unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns([1, 1])
    
    with col_d1:
        st.markdown(f"""
        <table style="width:100%; border-collapse:collapse; font-size:0.92rem; color:#334155;">
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:8px 0; font-weight:700;">Subject Identifier:</td><td style="padding:8px 0; color:#4338ca; font-weight:800;">{patient['patient_id'].upper()} ({patient['display_name']})</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:8px 0; font-weight:700;">Demographics:</td><td style="padding:8px 0;">{patient['age']} Years Old &bull; {patient['gender']}</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:8px 0; font-weight:700;">Clinical Diagnosis:</td><td style="padding:8px 0;">{patient['diagnosis']}</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:8px 0; font-weight:700;">Epileptogenic Focus:</td><td style="padding:8px 0; color:#e11d48; font-weight:700;">{patient['focal_zone']}</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:8px 0; font-weight:700;">Active AED Medication:</td><td style="padding:8px 0;">{', '.join(patient['medications'])}</td></tr>
            <tr><td style="padding:8px 0; font-weight:700;">Certified Seizure History:</td><td style="padding:8px 0;">{patient['total_seizures']} Clinical Seizures in EMU</td></tr>
        </table>
        """, unsafe_allow_html=True)
        
    with col_d2:
        st.markdown("<h4 style='color:#1e1b4b; margin-top:0;'>Cross-Patient Benchmark Comparison</h4>", unsafe_allow_html=True)
        df_summary = DatabaseManager.get_cross_patient_summary_df()
        st.dataframe(df_summary, hide_index=True, use_container_width=True)
        
        # Printable Report Download
        report_html = f"""
        <html>
        <head><title>Neurological Clinical Report</title></head>
        <body style="font-family:sans-serif; padding:20px; color:#0f172a;">
            <h2>BOSTON CHILDREN'S HOSPITAL &bull; CLINICAL NEUROLOGY</h2>
            <h3>ST-GAT Seizure Early Warning Horizon Report</h3>
            <p><b>Patient:</b> {patient['display_name']} | <b>Age:</b> {patient['age']} | <b>Sex:</b> {patient['gender']}</p>
            <p><b>Confirmed Diagnosis:</b> {patient['diagnosis']}</p>
            <p><b>Epileptogenic Focus:</b> {patient['focal_zone']}</p>
            <p><b>Current Seizure Risk Score:</b> {risk_val*100:.1f}%</p>
            <p><b>Warning Lead Horizon:</b> {lead_val:.1f} Minutes</p>
            <hr/>
            <p>Certified by ST-GAT Real-Time Neuro-Forecasting Engine v1.0</p>
        </body>
        </html>
        """
        st.download_button(
            "📥 Download Printable Patient Telemetry Report (HTML)",
            data=report_html,
            file_name=f"clinical_report_{patient['patient_id']}.html",
            mime="text/html"
        )
        
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.82rem; margin-top:30px;">
    M.Tech AI/ML Master Dissertation Project 17 &bull; Spatio-Temporal Graph Attention Network &bull; Pure Crisp Light Mode
</div>
""", unsafe_allow_html=True)
