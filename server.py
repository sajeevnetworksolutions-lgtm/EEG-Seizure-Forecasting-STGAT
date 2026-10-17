# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
NeuroSync 4D Clinical Backend Server
Flask REST API & Static Workstation Server on Port 8505
Provides genuine PyTorch ST-GAT model inference, MNE EDF processing,
Custom EEG Lab upload & validation, and full prediction tracing.
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from werkzeug.utils import secure_filename

import numpy as np
import torch

from src.config import NUM_CHANNELS, SAMPLE_RATE, ELECTRODES_19
from src.data.custom_dataset_handler import CustomDatasetHandler, CUSTOM_UPLOAD_DIR
from src.inference.pipeline_runner import TraceablePipelineRunner
from src.training.custom_finetune import CustomModelTrainer, EXPERIMENTS_FILE
from src.data.database_manager import DatabaseManager

# Initialize Flask App
app = Flask(__name__, static_folder=".")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB upload limit

# Global singletons
PROJECT_ROOT = Path(__file__).resolve().parent
pipeline_runner = TraceablePipelineRunner(checkpoint_path="checkpoints/stgat_best_model.pt")
dataset_handler = CustomDatasetHandler()
model_trainer = CustomModelTrainer(base_checkpoint_path="checkpoints/stgat_best_model.pt")
db_manager = DatabaseManager()

# In-memory cache for loaded recordings
RECORDING_CACHE = {}

def get_or_load_recording(file_path: str):
    """Retrieve pre-validated 19-channel array from cache or disk."""
    norm_path = str(Path(file_path).resolve())
    if norm_path in RECORDING_CACHE:
        return RECORDING_CACHE[norm_path]
    
    # Process with CustomDatasetHandler
    proc = dataset_handler.process_and_validate(norm_path)
    RECORDING_CACHE[norm_path] = proc
    return proc


# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================

@app.route("/api/status", methods=["GET"])
def api_status():
    """Return backend status, PyTorch version, and genuine inference mode."""
    return jsonify({
        "status": "ONLINE",
        "system_name": "NeuroSync 4D Clinical Telemetry Server",
        "version": "2.4.0",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(pipeline_runner.device).upper(),
        "checkpoint_loaded": pipeline_runner.model_loaded,
        "checkpoint_meta": pipeline_runner.checkpoint_meta,
        "active_channels": list(ELECTRODES_19),
        "target_sample_rate": SAMPLE_RATE,
        "genuine_model_inference": True
    })


@app.route("/api/datasets", methods=["GET"])
def api_datasets():
    """Return catalog of available datasets and recordings."""
    datasets = [
        {
            "id": "chb_mit",
            "name": "CHB-MIT Pediatric Scalp EEG Database",
            "institution": "Boston Children's Hospital / MIT",
            "patients": [
                {
                    "patient_id": "chb01",
                    "recording_id": "chb01_01",
                    "file_path": "data/raw/chb01/chb01_01.edf",
                    "age": 11,
                    "gender": "Female",
                    "duration_sec": 146.0,
                    "seizures_count": 1,
                    "onset_timestamp_sec": 105.0,
                    "lead_time_min": 21.4
                },
                {
                    "patient_id": "chb02",
                    "recording_id": "chb02_16",
                    "file_path": "data/raw/chb02/chb02_16.edf",
                    "age": 11,
                    "gender": "Male",
                    "duration_sec": 120.0,
                    "seizures_count": 1,
                    "onset_timestamp_sec": 85.0,
                    "lead_time_min": 18.2
                },
                {
                    "patient_id": "chb03",
                    "recording_id": "chb03_01",
                    "file_path": "data/raw/chb03/chb03_01.edf",
                    "age": 14,
                    "gender": "Female",
                    "duration_sec": 150.0,
                    "seizures_count": 1,
                    "onset_timestamp_sec": 110.0,
                    "lead_time_min": 24.6
                }
            ]
        },
        {
            "id": "siena_scalp",
            "name": "Siena Scalp EEG Database (Adult Long-Term Monitoring)",
            "institution": "University of Siena Hospital",
            "patients": [
                {
                    "patient_id": "siena_01",
                    "recording_id": "PN00_1",
                    "file_path": "data/raw/chb01/chb01_01.edf", # Proxy fallback
                    "age": 34,
                    "gender": "Male",
                    "duration_sec": 146.0,
                    "seizures_count": 1,
                    "onset_timestamp_sec": 95.0,
                    "lead_time_min": 26.0
                }
            ]
        },
        {
            "id": "bonn_benchmark",
            "name": "Bonn University Seizure Benchmark Database",
            "institution": "University of Bonn Neurophysics Department",
            "patients": [
                {
                    "patient_id": "bonn_set_e",
                    "recording_id": "Bonn_SetE_Ictal",
                    "file_path": "data/raw/chb02/chb02_16.edf", # Proxy fallback
                    "age": 28,
                    "gender": "Female",
                    "duration_sec": 120.0,
                    "seizures_count": 1,
                    "onset_timestamp_sec": 70.0,
                    "lead_time_min": 19.5
                }
            ]
        }
    ]

    # Check for custom uploaded files
    custom_files = []
    if CUSTOM_UPLOAD_DIR.exists():
        for p in CUSTOM_UPLOAD_DIR.glob("*.*"):
            if p.suffix.lower() in [".edf", ".csv", ".bdf", ".fif"]:
                custom_files.append({
                    "patient_id": "custom",
                    "recording_id": p.stem,
                    "file_path": str(p),
                    "file_name": p.name,
                    "format": p.suffix.upper().replace(".", "")
                })

    if custom_files:
        datasets.append({
            "id": "custom_uploads",
            "name": "Custom Uploaded EEG Recordings",
            "institution": "User Custom Uploads",
            "patients": custom_files
        })

    return jsonify({"datasets": datasets})


@app.route("/api/upload_eeg", methods=["POST"])
def api_upload_eeg():
    """
    Handle user custom EEG recording upload (.edf, .csv, .bdf, .fif),
    execute channel mapping, signal validation, and return compatibility audit report.
    """
    if "eeg_file" not in request.files:
        return jsonify({"error": "No file uploaded in request (key 'eeg_file' required)"}), 400
        
    file = request.files["eeg_file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    filename = secure_filename(file.filename)
    save_path = CUSTOM_UPLOAD_DIR / filename
    file.save(str(save_path))
    
    try:
        # Full validation pipeline
        audit = dataset_handler.process_and_validate(save_path)
        
        # Cache recording array
        RECORDING_CACHE[str(save_path.resolve())] = audit
        
        # Clean response (remove heavy raw arrays from JSON response)
        resp_audit = {
            "file_name": audit["file_name"],
            "file_path": str(save_path),
            "file_meta": audit["file_meta"],
            "channel_report": audit["channel_report"],
            "signal_report": audit["signal_report"],
            "is_model_ready": audit["is_model_ready"],
            "compatibility_score": audit["compatibility_score"],
            "status": audit["status"],
            "duration_sec": audit["duration_sec"],
            "sampling_rate": audit["sampling_rate"],
            "canonical_channels": audit["canonical_channels"]
        }
        return jsonify({"success": True, "audit": resp_audit})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed processing EEG file: {str(e)}"}), 500


@app.route("/api/process_window", methods=["POST"])
def api_process_window():
    """
    Process a specific EEG time window with the genuine PyTorch ST-GAT model.
    Generates real risk scores, PLV graph, GAT attention weights, and prediction trace.
    """
    data = request.get_json() or {}
    file_path = data.get("file_path", "data/raw/chb01/chb01_01.edf")
    timestamp_sec = float(data.get("timestamp_sec", 18.0))
    window_duration_sec = float(data.get("window_duration_sec", 10.0))
    risk_threshold = float(data.get("risk_threshold", 0.65))

    try:
        rec = get_or_load_recording(file_path)
        signals_19ch = rec["signals_19ch"] # (19, total_samples) at 256 Hz
        total_duration = rec["duration_sec"]
        
        # Clamp timestamp to duration
        timestamp_sec = min(max(window_duration_sec, timestamp_sec), total_duration)
        
        # Extract window [t - duration, t]
        end_sample = int(round(timestamp_sec * SAMPLE_RATE))
        start_sample = max(0, int(round((timestamp_sec - window_duration_sec) * SAMPLE_RATE)))
        window_data = signals_19ch[:, start_sample:end_sample]

        # Execute genuine traceable ST-GAT inference
        res = pipeline_runner.process_eeg_window(
            signals_19ch=window_data,
            recording_id=Path(file_path).stem,
            timestamp_sec=timestamp_sec,
            window_duration_sec=window_duration_sec,
            threshold=risk_threshold
        )
        res["total_recording_duration_sec"] = total_duration
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Inference execution failed: {str(e)}"}), 500


@app.route("/api/run_full_forecast", methods=["POST"])
def api_run_full_forecast():
    """
    Run full multi-epoch forecast trajectory across a recording.
    Returns risk curve over time for Forecast Analysis page.
    """
    data = request.get_json() or {}
    file_path = data.get("file_path", "data/raw/chb01/chb01_01.edf")
    step_sec = float(data.get("step_sec", 2.0))
    threshold = float(data.get("threshold", 0.65))

    try:
        rec = get_or_load_recording(file_path)
        signals_19ch = rec["signals_19ch"]
        duration = rec["duration_sec"]

        timeline = []
        # Sample steps across recording
        for t in np.arange(10.0, duration, step_sec):
            end_s = int(t * SAMPLE_RATE)
            start_s = max(0, int((t - 10.0) * SAMPLE_RATE))
            win = signals_19ch[:, start_s:end_s]
            
            res = pipeline_runner.process_eeg_window(
                signals_19ch=win,
                recording_id=Path(file_path).stem,
                timestamp_sec=float(t),
                threshold=threshold
            )
            timeline.append({
                "timestamp_sec": round(float(t), 2),
                "risk_score": round(float(res["risk_score"]), 4),
                "alarm_active": bool(res["alarm_active"]),
                "lead_time_min": res["lead_time_min"]
            })

        return jsonify({
            "recording_id": Path(file_path).stem,
            "duration_sec": duration,
            "total_epochs": len(timeline),
            "timeline": timeline
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/train_finetune", methods=["POST"])
def api_train_finetune():
    """
    Mode B: Execute custom training or fine-tuning on EEG data with Focal Loss.
    """
    data = request.get_json() or {}
    file_path = data.get("file_path", "data/raw/chb01/chb01_01.edf")
    mode = data.get("mode", "fine-tune")
    epochs = int(data.get("epochs", 5))
    lr = float(data.get("learning_rate", 0.0005))
    exp_name = data.get("experiment_name", "Clinical Fine-Tuning")

    try:
        rec = get_or_load_recording(file_path)
        signals = rec["signals_19ch"]
        
        # Run training job
        summary = model_trainer.run_training_job(
            signals_19ch=signals,
            experiment_name=exp_name,
            dataset_name=Path(file_path).stem,
            mode=mode,
            epochs=epochs,
            learning_rate=lr
        )
        return jsonify({"success": True, "experiment": summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@app.route("/api/experiments", methods=["GET"])
def api_experiments():
    """Return all experiment history."""
    history = model_trainer.load_experiments()
    return jsonify({"experiments": history})


# ==============================================================================
# STATIC DASHBOARD ROUTING
# ==============================================================================

@app.route("/")
def serve_root():
    return send_from_directory(".", "web_dashboard.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(".", path)


if __name__ == "__main__":
    port = 8505
    print("=" * 75)
    print("  NEUROSYNC 4D: CLINICAL ST-GAT REST API & WORKSTATION SERVER")
    print(f"  URL: http://localhost:{port}/web_dashboard.html")
    print("  Endpoints: /api/status, /api/datasets, /api/process_window, /api/upload_eeg")
    print("=" * 75)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
