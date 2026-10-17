# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
CHB-MIT Clinical Seizure Summary Annotation Parser
"""

from pathlib import Path
import pandas as pd
import re

def parse_summary_file(summary_path: str | Path):
    """
    Parse a single CHB-MIT patient summary text file (e.g. chb01-summary.txt).
    
    Returns
    -------
    records : list of dicts with file names, seizure counts, start/end timestamps.
    """
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return []

    patient_id = summary_path.parent.name if "chb" in summary_path.parent.name else summary_path.stem.split("-")[0]
    
    records = []
    current_file = None
    num_seizures = 0
    current_seizures = []
    
    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # File Name line
        if line.lower().startswith("file name:"):
            # Save previous file if exists
            if current_file is not None:
                records.append({
                    "patient_id": patient_id,
                    "recording_id": current_file,
                    "num_seizures": num_seizures,
                    "seizures": current_seizures
                })
            current_file = line.split(":")[-1].strip()
            num_seizures = 0
            current_seizures = []
            
        elif "number of seizures in file:" in line.lower():
            try:
                num_seizures = int(line.split(":")[-1].strip())
            except ValueError:
                num_seizures = 0
                
        elif "seizure" in line.lower() and "start time:" in line.lower():
            # Extract number of seconds
            match = re.search(r"(\d+)\s*seconds", line, re.IGNORECASE)
            if match:
                start_sec = int(match.group(1))
                current_seizures.append({"start_sec": start_sec, "end_sec": None})
                
        elif "seizure" in line.lower() and "end time:" in line.lower():
            match = re.search(r"(\d+)\s*seconds", line, re.IGNORECASE)
            if match and current_seizures:
                end_sec = int(match.group(1))
                current_seizures[-1]["end_sec"] = end_sec

    # Append last record
    if current_file is not None:
        records.append({
            "patient_id": patient_id,
            "recording_id": current_file,
            "num_seizures": num_seizures,
            "seizures": current_seizures
        })
        
    return records

def export_master_annotation_table(raw_dir: str | Path, output_csv_path: str | Path):
    """
    Search for all summary files in raw_dir and export a flattened master CSV.
    """
    raw_dir = Path(raw_dir)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_summary_files = list(raw_dir.glob("**/chb*-summary.txt"))
    rows = []
    
    for sum_file in all_summary_files:
        parsed_records = parse_summary_file(sum_file)
        for rec in parsed_records:
            if rec["num_seizures"] == 0:
                rows.append({
                    "patient_id": rec["patient_id"],
                    "recording_id": rec["recording_id"],
                    "has_seizure": 0,
                    "seizure_idx": 0,
                    "seizure_start_s": None,
                    "seizure_end_s": None,
                    "duration_s": 0
                })
            else:
                for s_idx, sz in enumerate(rec["seizures"], 1):
                    start = sz.get("start_sec", 0)
                    end = sz.get("end_sec", start + 30)
                    rows.append({
                        "patient_id": rec["patient_id"],
                        "recording_id": rec["recording_id"],
                        "has_seizure": 1,
                        "seizure_idx": s_idx,
                        "seizure_start_s": start,
                        "seizure_end_s": end,
                        "duration_s": (end - start) if (end and start) else 0
                    })
                    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    return df
