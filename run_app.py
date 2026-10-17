# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Launcher Script for Standalone HTML5/JS Clinical Neuro-Telemetry Workstation
Powered by Flask REST API backend + Genuine PyTorch ST-GAT Model Pipeline
"""

import sys
import webbrowser
import time
import threading
from pathlib import Path

def launch_dashboard(port=8505):
    project_root = Path(__file__).resolve().parent
    url = f"http://localhost:{port}/web_dashboard.html"
    
    print("=" * 75)
    print("  LAUNCHING NEUROSYNC 4D: CLINICAL ST-GAT TELEMETRY WORKSTATION")
    print("  Connected to genuine PyTorch ST-GAT Deep Inference Engine & REST API")
    print(f"  URL: {url}")
    print("=" * 75)
    
    # Open browser after slight delay to ensure server started
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    
    # Import and run server
    from server import app
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

if __name__ == "__main__":
    launch_dashboard()
