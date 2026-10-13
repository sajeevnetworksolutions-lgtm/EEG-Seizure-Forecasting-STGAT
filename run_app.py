"""
Launcher Script for Standalone HTML5/JS Clinical Neuro-Telemetry Workstation
Zero Streamlit Dependencies - Full Hospital GUI Console
"""

import sys
import subprocess
import webbrowser
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

def launch_dashboard(port=8505):
    project_root = Path(__file__).resolve().parent
    url = f"http://localhost:{port}/web_dashboard.html"
    
    print("=" * 75)
    print("  LAUNCHING NEUROSYNC 4D: CLINICAL ST-GAT TELEMETRY WORKSTATION")
    print(f"  URL: {url}")
    print("=" * 75)
    
    webbrowser.open(url)
    
    # Run HTTP Server in project root
    import os
    os.chdir(str(project_root))
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"Server active on port {port}. Press Ctrl+C to terminate.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nWorkstation server stopped.")

if __name__ == "__main__":
    launch_dashboard()
