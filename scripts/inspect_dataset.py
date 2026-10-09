"""
Dataset Inspection CLI Script
"""

import sys
from pathlib import Path
import yaml
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.parse_annotations import export_master_annotation_table

def inspect_dataset_cli():
    print("=" * 70)
    print("        CHB-MIT SCALP EEG DATASET INSPECTION & SUMMARY")
    print("=" * 70)
    
    config_path = PROJECT_ROOT / "configs" / "data.yaml"
    with open(config_path, "r") as f:
        data_cfg = yaml.safe_load(f)
        
    raw_dir = PROJECT_ROOT / data_cfg["dataset"]["raw_dir"]
    manifest_dir = PROJECT_ROOT / data_cfg["dataset"]["manifest_dir"]
    
    print(f"• Raw Dataset Directory:       {raw_dir.resolve()}")
    print(f"• Target Standard Montage:     19 Channels (10-20 International System)")
    print(f"• Nominal Sampling Rate:       {data_cfg['channels']['sampling_rate']} Hz")
    
    # Check for available EDF recordings
    edf_files = list(raw_dir.glob("**/*.edf"))
    print(f"• Local EDF Files Found:       {len(edf_files)}")
    
    if edf_files:
        for edf in edf_files[:5]:
            sz_mb = edf.stat().st_size / 1e6
            print(f"  - {edf.name} ({sz_mb:.2f} MB)")
            
    # Export / Parse annotations
    annot_csv = manifest_dir / "seizure_annotations.csv"
    if raw_dir.exists():
        df_annots = export_master_annotation_table(raw_dir, annot_csv)
        print(f"\n• Parsed Seizure Annotations exported to: {annot_csv.resolve()}")
        if not df_annots.empty:
            print(f"  - Total Annotation Entries: {len(df_annots)}")
            print(f"  - Verified Seizure Events:  {df_annots['has_seizure'].sum()}")
            
    print("=" * 70)

if __name__ == "__main__":
    inspect_dataset_cli()
