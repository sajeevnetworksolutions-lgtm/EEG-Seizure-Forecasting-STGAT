# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
ST-GAT Real-Time Epileptic Seizure Forecasting Package
"""
import sys
import os
from pathlib import Path

# Windows DLL directory fix for PyTorch c10.dll loading
if sys.platform == "win32":
    try:
        import site
        for sp in site.getsitepackages():
            torch_lib = Path(sp) / "torch" / "lib"
            if torch_lib.exists():
                os.add_dll_directory(str(torch_lib))
    except Exception:
        pass

# Polyfill np.dtypes.StringDType for MNE with NumPy < 2.0
try:
    import numpy as np
    if hasattr(np, "dtypes") and not hasattr(np.dtypes, "StringDType"):
        np.dtypes.StringDType = np.dtype("U")
except Exception:
    pass

__version__ = "1.0.0"

