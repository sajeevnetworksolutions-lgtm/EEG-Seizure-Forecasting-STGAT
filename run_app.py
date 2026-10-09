"""
Launcher Script for Streamlit Hospital Monitoring Dashboard
"""

import sys
import subprocess
from pathlib import Path

def launch_dashboard():
    project_root = Path(__file__).resolve().parent
    home_app = project_root / "app" / "Home.py"
    
    print("=" * 70)
    print("  LAUNCHING ST-GAT CLINICAL EEG SEIZURE FORECASTING DASHBOARD")
    print(f"  Target File: {home_app.resolve()}")
    print("=" * 70)
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(home_app)]
    subprocess.run(cmd)

if __name__ == "__main__":
    launch_dashboard()
